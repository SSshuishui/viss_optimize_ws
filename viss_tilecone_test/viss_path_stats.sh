# Compile example:
nvcc -O3 --use_fast_math -lineinfo -std=c++17 -Xcompiler -fopenmp viss_path_stats.cu -o viss_path_stats

# 10 MHz full online statistics
# ./viss_path_stats \
#   --btag=10M \
#   --nside=4096 \
#   --day_start=1 \
#   --day_count=1 \
#   --segs=10 \
#   --gpu=0 \
#   --viss_tile_pix=256 \
#   --stats_tile_stride=1 \
#   --stats_max_tiles=0

# 30 MHz suggested sampled statistics
./viss_path_stats \
  --btag=30M \
  --nside=16384 \
  --day_start=1 \
  --day_count=1 \
  --segs=10 \
  --gpu=0,1,2,3 \
  --viss_tile_pix=256 \
  --stats_tile_stride=2 

  # --stats_max_tiles=200000
