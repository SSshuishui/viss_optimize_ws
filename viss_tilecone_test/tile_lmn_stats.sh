# Build
# nvcc -O3 -std=c++17 -arch=sm_86 tile_lmn_stats.cu -o tile_lmn_stats

# 10 MHz full statistics
# ./tile_lmn_stats \
#   --gpu=0 \
#   --nside=4096 \
#   --tiles=64,128,256,512 \
#   --chunk_pix=1048576

# 30 MHz quick sample (stop after 200k tiles per size)
CUDA_VISIBLE_DEVICES=2 ./tile_lmn_stats \
  --gpu=0 \
  --nside=16384 \
  --tiles=64,128,256,512 \
  --chunk_pix=1048576

#   --max_tiles=200000 \
