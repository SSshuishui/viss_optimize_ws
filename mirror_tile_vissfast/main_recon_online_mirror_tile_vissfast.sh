echo "========================================"
echo "生成任务启动: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

START_TIME=$(date +%s)

# nvcc -O3 --use_fast_math -lineinfo -std=c++17 -Xcompiler -fopenmp main_recon_online_mirror_tile_vissfast.cu -o main_recon_online_mirror_tile_vissfast

# 1 MHz check
# ./main_recon_online_mirror_tile_vissfast \
#   --btag=1M \
#   --nside=512 \
#   --day=1 \
#   --segs=10 \
#   --blockage=1 \
#   --occ_mode=2 \
#   --sky_dir=../earth_1Mhz \
#   --out_dir=../out1M_online_mirror_tile_vissfast/ \
#   --gpus=0,1 \
#   --gen_gpu_index=0 \
#   --B_mode=txt \
#   --C_mode=bin \
#   --orbit_seed=42 \


# 10 MHz check
./main_recon_online_mirror_tile_vissfast \
  --btag=10M \
  --nside=4096 \
  --day_start=1 \
  --day_count=1 \
  --segs=10 \
  --blockage=1 \
  --occ_mode=2 \
  --sky_dir=../earth_10Mhz \
  --out_dir=../out10M_vissfast/ \
  --gpus=0,1 \
  --gen_gpu_index=0 \
  --B_mode=bin \
  --C_mode=bin \
  --orbit_seed=42 

# 30 MHz production
# ./main_recon_online_mirror_tile_vissfast \
#   --btag=30M \
#   --nside=16384 \
#   --day_start=1 \
#   --day_count=1 \
#   --segs=30 \
#   --blockage=1 \
#   --occ_mode=2 \
#   --sky_dir=../earth_30Mhz \
#   --out_dir=../out30M_online_mirror_tile_vissfast/ \
#   --gpus=4,5,6,7 \
#   --gen_gpu_index=0 \
#   --B_mode=bin \
#   --C_mode=bin \
#   --orbit_seed=42


EXIT_CODE=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
H=$((ELAPSED / 3600))
M=$(((ELAPSED % 3600) / 60))
S=$((ELAPSED % 60))

echo ""
echo "========================================"
echo "生成任务结束: $(date '+%Y-%m-%d %H:%M:%S')"
echo "总耗时: ${H}h ${M}m ${S}s"
echo "退出码: ${EXIT_CODE}"
echo "========================================"

exit ${EXIT_CODE}