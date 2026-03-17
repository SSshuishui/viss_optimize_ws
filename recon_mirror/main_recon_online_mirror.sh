# nvcc -O3 --use_fast_math -lineinfo -std=c++17 -Xcompiler -fopenmp -arch=sm_86 main_recon_online_mirror.cu -o main_recon_online_mirror

# 1 MHz check
# ./main_recon_online_mirror \
#   --btag=1M \
#   --nside=512 \
#   --day=1 \
#   --segs=1 \
#   --blockage=1 \
#   --occ_mode=2 \
#   --sky_dir=../earth_1Mhz \
#   --out_dir=../out1M_online_recon_mirror/ \
#   --gpus=0 \
#   --gen_gpu_index=0 \
#   --B_mode=txt \
#   --C_mode=bin \
#   --orbit_seed=42 \
#   --write_viss=1 \
#   --write_baseline_txt=1

# 10 MHz check
./main_recon_online_mirror \
  --btag=10M \
  --nside=4096 \
  --day=1 \
  --segs=10 \
  --blockage=1 \
  --occ_mode=2 \
  --sky_dir=../earth_10Mhz \
  --out_dir=../out10M_online_recon_mirror/ \
  --gpus=1,3 \
  --gen_gpu_index=0 \
  --B_mode=txt \
  --C_mode=bin \
  --orbit_seed=42

# 30 MHz production
# ./main_recon_online_mirror \
#   --btag=30M \
#   --nside=16384 \
#   --day_start=1 \
#   --day_count=1 \
#   --segs=30 \
#   --blockage=1 \
#   --occ_mode=2 \
#   --sky_dir=../earth_30Mhz \
#   --out_dir=../out30M_online_recon_mirror/ \
#   --gpus=0,1,2,3 \
#   --gen_gpu_index=0 \
#   --B_mode=bin \
#   --C_mode=bin \
#   --orbit_seed=42
