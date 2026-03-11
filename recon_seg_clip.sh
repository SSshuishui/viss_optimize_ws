./recon_seg_clip \
  --btag=10M --nside=4096 --day=1 --segs=10 \
  --blockage=1 --occ_mode=2 \
  --in_dir=./earth_10Mhz \
  --sky_dir=./earth_10Mhz \
  --out_dir=./out10M_recon_seg/ \
  --gpus=0,1,2,3 --uvw_max=4500000 --B_clip=500000 --write_viss=1