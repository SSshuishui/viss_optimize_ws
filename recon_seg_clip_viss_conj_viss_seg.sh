./recon_seg_clip_viss_conj_viss_seg \
  --btag=10M --nside=4096 --day=1 --segs=10 \
  --blockage=1 --occ_mode=2 \
  --in_dir=./earth_10Mhz \
  --sky_dir=./earth_10Mhz \
  --out_dir=./out10M_recon_seg_viss_conj_viss_seg/ \
  --gpus=0,1,2 --uvw_max=4500000 --B_clip=500000 --write_viss=1