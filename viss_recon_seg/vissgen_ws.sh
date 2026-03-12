./vissgen_ws \
  --btag=10M --nside=4096 --day=1 \
  --blockage=1 --occ_mode=2 \
  --in_dir=./earth_10Mhz_cuda \
  --sky_dir=./earth_10Mhz \
  --out_dir=./out10M_fullpipe_334/ \
  --gpus=1,2,3 \
  --uvw_max=4500000 \
  --B_clip=500000