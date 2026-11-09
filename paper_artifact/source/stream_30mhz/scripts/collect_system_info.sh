#!/usr/bin/env bash
set -u
OUT_DIR=${1:-system_info_4090}
mkdir -p "$OUT_DIR"

uname -a > "$OUT_DIR/uname.txt" 2>&1 || true
lscpu > "$OUT_DIR/lscpu.txt" 2>&1 || true
free -h > "$OUT_DIR/free_h.txt" 2>&1 || true
lsblk -o NAME,MODEL,SIZE,ROTA,TYPE,FSTYPE,MOUNTPOINT > "$OUT_DIR/lsblk.txt" 2>&1 || true
nvcc --version > "$OUT_DIR/nvcc_version.txt" 2>&1 || true
gcc --version > "$OUT_DIR/gcc_version.txt" 2>&1 || true
nvidia-smi -q > "$OUT_DIR/nvidia_smi_q.txt" 2>&1 || true
nvidia-smi topo -m > "$OUT_DIR/nvidia_smi_topo.txt" 2>&1 || true
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version,pci.bus_id --format=csv > "$OUT_DIR/gpu_inventory.csv" 2>&1 || true

echo "[OK] System information written to $OUT_DIR"
