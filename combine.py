import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import glob
import os
import re
from pathlib import Path

def find_and_sort_files(directory="./out30M_streamonly_seg10/", pattern="C*day30M.bin"):
    """
    查找并排序所有匹配的文件
    返回按数字n排序的文件路径列表
    """
    search_pattern = os.path.join(directory, pattern)
    files = glob.glob(search_pattern)
    
    # 提取数字n并排序
    def extract_number(filepath):
        filename = os.path.basename(filepath)
        # 匹配 Cnday30M.bin 中的 n
        match = re.match(r'C(\d+)day30M\.bin', filename)
        if match:
            return int(match.group(1))
        return float('inf')  # 不符合格式的文件排在最后
    
    files.sort(key=extract_number)
    return files

def load_healpix_bin(filepath, dtype=np.float32):
    """
    加载单个bin文件
    """
    return np.fromfile(filepath, dtype=dtype)

def save_healpix_bin(data, filepath, dtype=np.float32):
    """
    保存数据到bin文件
    """
    data.astype(dtype).tofile(filepath)
    print(f"已保存到: {filepath}")

def accumulate_healpix_files(directory="./out30M_streamonly_seg10/", 
                             pattern="C*day30M.bin",
                             output_file=None,
                             previous_sum_file=None,
                             dtype=np.float32):
    """
    累加所有Healpix bin文件
    
    Parameters:
    -----------
    directory : str
        文件所在目录
    pattern : str
        文件匹配模式
    output_file : str
        输出文件名，默认为 accumulated_sum.bin
    previous_sum_file : str
        之前保存的累加和文件，如果提供则从此继续累加
    dtype : numpy dtype
        数据类型
    """
    
    # 查找所有文件
    files = find_and_sort_files(directory, pattern)
    
    if not files:
        print(f"未找到匹配 {pattern} 的文件在 {directory}")
        return None
    
    print(f"找到 {len(files)} 个文件:")
    for f in files:
        print(f"  - {os.path.basename(f)}")
    
    # 初始化累加和
    if previous_sum_file and os.path.exists(previous_sum_file):
        print(f"\n加载之前的累加和: {previous_sum_file}")
        accumulated_sum = load_healpix_bin(previous_sum_file, dtype)
        print(f"之前累加和的形状: {accumulated_sum.shape}")
        n_files_processed = 0  # 这里假设之前已经处理过，或者从metadata读取
    else:
        # 读取第一个文件获取形状
        print(f"\n读取第一个文件确定形状: {os.path.basename(files[0])}")
        first_data = load_healpix_bin(files[0], dtype)
        accumulated_sum = np.zeros_like(first_data)
        n_files_processed = 0
    
    # 累加所有文件
    print(f"\n开始累加...")
    for i, filepath in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] 处理: {os.path.basename(filepath)}")
        data = load_healpix_bin(filepath, dtype)
        
        # 检查维度匹配
        if data.shape != accumulated_sum.shape:
            raise ValueError(f"维度不匹配! {filepath}: {data.shape} vs {accumulated_sum.shape}")
        
        accumulated_sum += data
    
    print(f"\n累加完成! 共处理 {len(files)} 个文件")
    print(f"结果形状: {accumulated_sum.shape}")
    print(f"结果范围: [{accumulated_sum.min():.6f}, {accumulated_sum.max():.6f}]")
    print(f"结果均值: {accumulated_sum.mean():.6f}")
    
    # 保存结果
    if output_file is None:
        output_file = os.path.join(directory, "accumulated_sum.bin")
    
    save_healpix_bin(accumulated_sum, output_file, dtype)
    
    # 保存元数据（记录处理了多少文件等信息）
    metadata_file = output_file.replace('.bin', '_metadata.txt')
    with open(metadata_file, 'w') as f:
        f.write(f"Number of files processed: {len(files)}\n")
        f.write(f"Files:\n")
        for fp in files:
            f.write(f"  {fp}\n")
        f.write(f"\nOutput shape: {accumulated_sum.shape}\n")
        f.write(f"Data type: {dtype}\n")
        f.write(f"Min: {accumulated_sum.min()}\n")
        f.write(f"Max: {accumulated_sum.max()}\n")
        f.write(f"Mean: {accumulated_sum.mean()}\n")
    
    print(f"元数据已保存到: {metadata_file}")
    
    return accumulated_sum, output_file

def visualize_accumulated(data, nside=16384, title="Accumulated Healpix Map", nest=True):
    """
    可视化累加结果
    """
    hp.mollview(data, nest=nest, title=title)
    hp.graticule()
    plt.show()
    plt.savefig('accumulated_healpix_map.png')

# ==================== 使用示例 ====================

if __name__ == "__main__":
    
    # 示例1: 第一次运行，从头开始累加
    print("=" * 50)
    print("示例1: 从头开始累加所有文件")
    print("=" * 50)
    
    sum_data, output_path = accumulate_healpix_files(
        directory="./out30M_streamonly_seg10/",
        pattern="accumulated_*.bin",
        output_file="./out30M_streamonly/accumulated_450.bin",
        previous_sum_file=None  # 第一次运行，没有之前的累加和
    )

    # print("=" * 50)
    # print("示例2: 加载已有累加文件后，继续累加所有文件")
    # print("=" * 50)

    # sum_data, output_path = accumulate_healpix_files(
    #     directory="./out30M_streamonly_seg10/",
    #     pattern="day*_30M.bin",
    #     output_file="./out30M_streamonly_seg10/accumulated_1_110.bin",
    #     previous_sum_file="./out30M_streamonly_seg10/accumulated_1_109.bin"  # 加载之前的累加和
    # )
    
    # 可视化结果
    if sum_data is not None:
        visualize_accumulated(sum_data, title="Accumulated Sum of All C*day30M.bin files")
    
    