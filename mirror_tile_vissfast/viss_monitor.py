#!/usr/bin/env python3
"""
accumulator_daemon_with_generator.py
带生成器启动的累加守护进程
"""
import os
import time
import glob
import numpy as np
import re
import threading
import queue
import subprocess
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import healpy as hp

# ==================== 配置 ====================
WATCH_DIR = "../out10M_vissfast/"
FIGS_DIR = "../out10M_vissfast/figs2"
ACCUM_FILE = os.path.join(WATCH_DIR, "run_450_wait.bin")
PATTERN = "C*day10M.bin"
DTYPE = np.float32
NSIDE = 4096
NEST = True

CHECK_INTERVAL = 10
FILE_STABLE_TIME = 3.0
PROCESSING_TIMEOUT = 300
# =============================================

class AccumulatorDaemon:
    def __init__(self):
        self.processing_queue = queue.Queue()
        self.processed_files = set()
        self.accum_lock = threading.Lock()
        self.stats = {'processed': 0, 'failed': 0, 'total_size_gb': 0}
        self.running = True
        self.generator_process = None
        self.generator_done = False
        self.overall_start_time = None
        
    def ensure_dirs(self):
        os.makedirs(WATCH_DIR, exist_ok=True)
        os.makedirs(FIGS_DIR, exist_ok=True)
        
    def get_npix(self):
        return 12 * NSIDE ** 2
        
    def init_accum_file(self, npix):
        if not os.path.exists(ACCUM_FILE):
            print(f"[初始化] 创建累加文件: {npix:,} 像素 ({npix*4/1e9:.2f} GB)")
            zeros = np.zeros(npix, dtype=DTYPE)
            zeros.tofile(ACCUM_FILE)
            del zeros
        else:
            size = os.path.getsize(ACCUM_FILE)
            print(f"[恢复] 加载已有累加: {size/1e9:.2f} GB")
            
    def is_file_stable(self, filepath, wait_time=FILE_STABLE_TIME):
        try:
            size1 = os.path.getsize(filepath)
            if size1 == 0:
                return False
            time.sleep(wait_time)
            size2 = os.path.getsize(filepath)
            return size1 == size2
        except OSError:
            return False
            
    def extract_number(self, filepath):
        filename = os.path.basename(filepath)
        match = re.match(r'C(\d+)day10M\.bin', filename)
        return int(match.group(1)) if match else float('inf')
        
    def visualize(self, data, tag, title):
        try:
            plt.figure(figsize=(12, 7))
            hp.mollview(data, nest=NEST, title=title, unit="K", 
                       min=np.percentile(data, 1), max=np.percentile(data, 99))
            hp.graticule()
            plt.savefig(os.path.join(FIGS_DIR, f"{tag}.png"), 
                       dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"    [画图] {tag}.png 完成")
        except Exception as e:
            print(f"    [画图错误] {e}")
            
    def process_single_file(self, filepath):
        filename = os.path.basename(filepath)
        file_num = self.extract_number(filename)
        npix = self.get_npix()
        
        print(f"\n[{time.strftime('%H:%M:%S')}] 开始处理 #{self.stats['processed']+1}: {filename}")
        print(f"    文件大小: {os.path.getsize(filepath)/1e9:.2f} GB")
        start_time = time.time()
        
        try:
            accum = np.memmap(ACCUM_FILE, dtype=DTYPE, mode='r+', shape=(npix,))
            data = np.memmap(filepath, dtype=DTYPE, mode='r', shape=(npix,))
            
            print(f"    累加计算中...")
            with self.accum_lock:
                accum[:] += data[:]
                accum.flush()
            
            curr_max = accum.max()
            curr_min = accum.min()
            curr_mean = accum.mean()
            
            del data
            
            fig_title = f"C{file_num} | Mean:{curr_mean:.2e} Range:[{curr_min:.2e},{curr_max:.2e}]"
            self.visualize(accum, f"C{file_num}", fig_title)
            
            del accum
            
            os.remove(filepath)
            
            elapsed = time.time() - start_time
            self.stats['processed'] += 1
            self.stats['total_size_gb'] += 1
            
            print(f"    ✅ 完成 ({elapsed:.1f}s) | 累计处理: {self.stats['processed']} 个")
            print(f"    当前累加和: Mean={curr_mean:.6f}, Max={curr_max:.6f}")
            
            with open(ACCUM_FILE + ".log", "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | C{file_num} | "
                       f"time={elapsed:.1f}s | mean={curr_mean:.6f}\n")
                       
            return True
            
        except Exception as e:
            print(f"    ❌ 失败: {e}")
            self.stats['failed'] += 1
            return False
            
    def start_generator(self, script_path):
        """启动生成脚本子进程"""
        print(f"\n[生成器] 启动: {script_path}")
        print(f"[生成器] 工作目录: {os.getcwd()}")
        
        # 确保脚本存在
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"生成脚本不存在: {script_path}")
            
        # 使用 Popen 非阻塞启动，并捕获输出
        self.generator_process = subprocess.Popen(
            ['bash', script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=os.getcwd()
        )
        print(f"[生成器] PID: {self.generator_process.pid}")
        
    def monitor_generator_output(self):
        """在后台线程中实时读取生成器输出"""
        if self.generator_process is None:
            return
            
        try:
            for line in self.generator_process.stdout:
                line = line.rstrip()
                if line:
                    print(f"  [GEN] {line}")
        except Exception as e:
            print(f"[生成器输出错误] {e}")
            
    def wait_generator(self):
        """等待生成器结束并获取退出码"""
        if self.generator_process is None:
            self.generator_done = True
            return 0
            
        return_code = self.generator_process.wait()
        self.generator_done = True
        print(f"\n[生成器] 已结束，退出码: {return_code}")
        return return_code
                
    def scanner_thread(self):
        print(f"[扫描线程] 启动，监控: {WATCH_DIR}")
        
        while self.running:
            try:
                files = glob.glob(os.path.join(WATCH_DIR, PATTERN))
                files.sort(key=self.extract_number)
                
                for filepath in files:
                    if not self.running:
                        break
                        
                    if filepath == ACCUM_FILE:
                        continue
                        
                    if filepath in self.processed_files:
                        continue
                        
                    if not self.is_file_stable(filepath):
                        continue
                        
                    print(f"[扫描] 发现新文件: {os.path.basename(filepath)}")
                    self.processing_queue.put(filepath)
                    self.processed_files.add(filepath)
                    
                time.sleep(CHECK_INTERVAL)
                
            except Exception as e:
                print(f"[扫描错误] {e}")
                time.sleep(CHECK_INTERVAL)
                
    def worker_thread(self):
        print(f"[工作线程] 启动，准备处理...")
        
        while self.running:
            try:
                filepath = self.processing_queue.get(timeout=5)
                self.process_single_file(filepath)
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[工作错误] {e}")
                
    def format_elapsed(self, seconds):
        """格式化耗时"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}h {m}m {s}s"
        elif m > 0:
            return f"{m}m {s}s"
        else:
            return f"{s}s"
                
    def run(self, script_path=None):
        print("=" * 70)
        print("累加守护进程启动 (Generator Edition)")
        print("=" * 70)
        print(f"配置: 检查间隔={CHECK_INTERVAL}s, 文件稳定检测={FILE_STABLE_TIME}s")
        print("=" * 70)
        
        self.ensure_dirs()
        npix = self.get_npix()
        self.init_accum_file(npix)
        
        # 启动生成器（如果提供了脚本路径）
        gen_monitor = None
        if script_path:
            self.start_generator(script_path)
            # 后台线程读取生成器输出
            gen_monitor = threading.Thread(target=self.monitor_generator_output, daemon=True)
            gen_monitor.start()
        
        # 先处理目录中已存在的文件
        print("\n[初始化] 扫描已有文件...")
        existing = glob.glob(os.path.join(WATCH_DIR, PATTERN))
        for f in sorted(existing, key=self.extract_number):
            if f != ACCUM_FILE and self.is_file_stable(f, wait_time=1.0):
                self.processing_queue.put(f)
                self.processed_files.add(f)
                print(f"  排队: {os.path.basename(f)}")
                
        # 启动工作线程
        scanner = threading.Thread(target=self.scanner_thread, daemon=True)
        worker = threading.Thread(target=self.worker_thread, daemon=True)
        
        scanner.start()
        worker.start()
        
        # 记录整体开始时间
        self.overall_start_time = time.time()
        print(f"\n[运行中] 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"按 Ctrl+C 强制停止 | 生成器结束后自动检测完成并退出")
        print("-" * 70)
        
        try:
            while True:
                qsize = self.processing_queue.qsize()
                elapsed = time.time() - self.overall_start_time
                
                # 检查是否应该退出：生成器已结束 + 队列为空 + 没有新文件
                if self.generator_done and qsize == 0:
                    # 再等一个检查周期，确保没有遗漏
                    time.sleep(CHECK_INTERVAL + 2)
                    # 最终检查
                    final_check = glob.glob(os.path.join(WATCH_DIR, PATTERN))
                    has_new = False
                    for f in final_check:
                        if f != ACCUM_FILE and f not in self.processed_files and self.is_file_stable(f, wait_time=1.0):
                            has_new = True
                            self.processing_queue.put(f)
                            self.processed_files.add(f)
                            print(f"[收尾] 发现遗漏文件: {os.path.basename(f)}")
                    
                    if not has_new and self.processing_queue.qsize() == 0:
                        print(f"\n{'=' * 70}")
                        print("[完成] 生成器已结束，所有文件处理完毕")
                        break
                
                # 显示状态
                status = "运行中" if not self.generator_done else "等待收尾"
                print(f"\r[{status}] 已运行: {self.format_elapsed(elapsed)} | "
                      f"队列: {qsize} | 已处理: {self.stats['processed']} | "
                      f"已处理容量: {self.stats['total_size_gb']:.1f} GB", end="")
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\n\n[停止] 用户中断...")
            
        # ===== 清理和统计 =====
        self.running = False
        total_elapsed = time.time() - self.overall_start_time
        
        print(f"\n{'=' * 70}")
        print("最终统计")
        print(f"{'=' * 70}")
        print(f"  总运行时间: {self.format_elapsed(total_elapsed)}")
        print(f"  成功处理: {self.stats['processed']} 个文件")
        print(f"  失败: {self.stats['failed']} 个")
        print(f"  总处理数据: {self.stats['total_size_gb']:.1f} GB")
        print(f"  平均每个文件: {total_elapsed/max(self.stats['processed'],1):.1f}s")
        print(f"累加文件: {ACCUM_FILE}")
        print(f"日志: {ACCUM_FILE}.log")
        print(f"图片: {FIGS_DIR}/")
        
        # 等待工作线程完成当前任务
        if worker.is_alive():
            print("\n等待当前处理完成...")
            worker.join(timeout=PROCESSING_TIMEOUT + 10)
            
        # 如果生成器还在运行，终止它
        if self.generator_process and self.generator_process.poll() is None:
            print("[清理] 终止生成器进程...")
            self.generator_process.terminate()
            try:
                self.generator_process.wait(timeout=5)
            except:
                self.generator_process.kill()
                
        print("守护进程已退出")

if __name__ == "__main__":
    # 用法: python accumulator_daemon_with_generator.py [生成脚本路径]
    # 如果不提供路径，则只作为纯监控进程运行
    script_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    daemon = AccumulatorDaemon()
    daemon.run(script_path=script_path)