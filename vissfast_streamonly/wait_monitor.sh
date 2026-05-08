#!/bin/bash
# wait_for_gpus_and_monitor.sh

SCRIPT_TO_RUN="bash main_recon_online_mirror_tile_vissfast_streamonly.sh"
CHECK_INTERVAL=30
GPUS_TO_CHECK="0,1"
MEM_THRESHOLD=500
UTIL_THRESHOLD=5

echo "[$(date)] 等待 GPU ${GPUS_TO_CHECK} 空闲..."

while true; do
    mapfile -t mem_used < <(nvidia-smi -i "$GPUS_TO_CHECK" --query-gpu=memory.used --format=csv,noheader,nounits)
    mapfile -t gpu_util < <(nvidia-smi -i "$GPUS_TO_CHECK" --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    
    idle_count=0
    gpu_indices=(${GPUS_TO_CHECK//,/ })
    
    for idx in "${!gpu_indices[@]}"; do
        mem="${mem_used[$idx]// /}"
        util="${gpu_util[$idx]// /}"
        if [ "$mem" -lt "$MEM_THRESHOLD" ] && [ "$util" -lt "$UTIL_THRESHOLD" ]; then
            ((idle_count++))
        fi
    done

    if [ "$idle_count" -eq "${#gpu_indices[@]}" ]; then
        echo "[$(date)] ✅ GPU 空闲，启动任务..."
        break
    fi
    sleep "$CHECK_INTERVAL"
done


cleanup() {
    echo "[$(date)] 主程序结束，等待守护进程完成剩余文件处理..."
    # 等待队列清空（最多等10分钟，因为处理需要时间）
    for i in {1..60}; do
        if ! kill -0 $MONITOR_PID 2>/dev/null; then
            echo "[$(date)] 守护进程已自动退出"
            break
        fi
        # 检查队列是否为空（通过日志或标记文件）
        if [ -f "../out30M_streamonly_seg10/running_accum.bin.log" ]; then
            # 简单等待固定时间，确保最后几个文件被处理
            sleep 10
        fi
    done
    
    # 如果还在运行，强制终止
    if kill -0 $MONITOR_PID 2>/dev/null; then
        echo "[$(date)] 强制停止守护进程"
        kill $MONITOR_PID
        sleep 2
    fi
    exit 0
}

trap cleanup EXIT INT TERM

echo "[$(date)] 启动主程序..."
eval "$SCRIPT_TO_RUN"