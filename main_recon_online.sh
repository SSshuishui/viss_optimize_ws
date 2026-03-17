# nvcc -O3 --use_fast_math -lineinfo -std=c++17 -Xcompiler -fopenmp -arch=sm_86 main_recon_online.cu -o main_recon_online

# ./main_recon_online \
#   --btag=1M \
#   --nside=512 \
#   --day=1 \
#   --segs=10 \
#   --blockage=1 \
#   --occ_mode=2 \
#   --sky_dir=./earth_1Mhz \
#   --out_dir=./out1M_online/ \
#   --gpus=0 \
#   --gen_gpu_index=0 \
#   --B_mode=txt \
#   --orbit_seed=42 \
#   --write_viss=1 \
#   --write_baseline_txt=1

./main_recon_online \
  --btag=10M \
  --nside=4096 \
  --day=1 \
  --segs=10 \
  --blockage=1 \
  --occ_mode=2 \
  --sky_dir=./earth_10Mhz \
  --out_dir=./out10M_online/ \
  --gpus=1,3 \
  --gen_gpu_index=0 \
  --B_mode=txt \
  --orbit_seed=42 \
  --write_viss=1 \
  --write_baseline_txt=1


# ./main_recon_online \
#   --btag=30M \
#   --nside=16384 \
#   --day=1 \
#   --segs=10 \
#   --blockage=1 \
#   --occ_mode=2 \
#   --sky_dir=./earth_30Mhz \
#   --out_dir=./out30M_online/ \
#   --gpus=0,1,2,3 \
#   --gen_gpu_index=0 \
#   --B_mode=bin \
#   --orbit_seed=42 