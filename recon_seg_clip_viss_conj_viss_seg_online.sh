./recon_seg_clip_viss_conj_viss_seg_online \
  --btag=10M --nside=4096 --day=1 --segs=10 \
  --blockage=0 --occ_mode=2 \
  --in_dir=./earth_10Mhz \
  --sky_dir=./earth_10Mhz \
  --out_dir=./out_ref_online_lmn/ \
  --gpus=0,1 \
  --uvw_max=4500000 \
  --write_viss=1 \
  --lmn_mode=online