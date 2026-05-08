# 4090 build
# nvcc -O3 --use_fast_math -lineinfo -std=c++17 -Xcompiler -fopenmp -arch=sm_89 main_recon_online_mirror_tile_vissfast_streamonly.cu -o main_recon_online_mirror_tile_vissfast_streamonly


# ./main_recon_online_mirror_tile_vissfast_streamonly \
#   --btag=10M \
#   --nside=4096 \
#   --day_start=1 \
#   --day_count=1 \
#   --segs=10 \
#   --blockage=1 \
#   --occ_mode=2 \
#   --sky_dir=../earth_10Mhz \
#   --out_dir=../out10M_streamonly/ \
#   --gpus=0,1 \
#   --gen_gpu_index=0 \
#   --B_mode=txt \
#   --C_mode=bin \
#   --orbit_seed=42 \
#   --viss_tile_pix=256 \
#   --sky_slab_pix=33554432


# Example: 30 MHz on 4x4090 with streamed sky slabs
./main_recon_online_mirror_tile_vissfast_streamonly \
  --btag=30M \
  --nside=16384 \
  --day_start=386 \
  --day_count=1 \
  --segs=10 \
  --blockage=1 \
  --occ_mode=2 \
  --sky_dir=../earth_30Mhz \
  --out_dir=../out30M_streamonly_seg10/ \
  --gpus=0,1 \
  --gen_gpu_index=0 \
  --B_mode=bin \
  --C_mode=bin \
  --orbit_seed=42 \
  --viss_tile_pix=256 \
  --sky_slab_pix=402653184
