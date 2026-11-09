
# 1. 使用nsys分析看整体
nsys profile -o pipeline_multigpu \
  --force-overwrite=true \
  --sample=none \
  --cpuctxsw=none \
  --trace=cuda \
  --cuda-memory-usage=false \
  ./shared_operator_plan_recon_fastpath \
  --btag=10M \
  --nside=4096 \
  --day_start=1 \
  --day_count=3 \
  --segs=10 \
  --sky_dir=../input_skymap/sky_sao/10M_n4096_ring_ecliptic/B_10M \
  --out_dir=./out10M/ \
  --gpus=1,3 \
  --gen_gpu_index=0 \
  --B_mode=bin \
  --C_mode=bin \
  --sky_order=ring \
  --orbit_seed=42 \
  --dcf_bin=dcf_mb_10M_days450_seed42.bin \
  --viss_tile_pix=256



# 2. 用 NCU 分析热点 kernel（单GPU）
# # 只分析 viss kernel
# ncu --set full \
#  --kernel-name "*viss_partial*" \
#  -o viss_analysis \
#  ./main_3d_viss_recon --gpus=0 ...

# # 只分析 recon kernel
# ncu --set full \
#  --kernel-name "*recon_3d*" \
#  -o recon_analysis \
#  ./main_3d_viss_recon --gpus=0 ...
