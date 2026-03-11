# 1M  512
# 10M 4096
# 30M 16384

./pix2fang_nest \
  --out_dir=./earth_1Mhz \
  --nside=512 \
  --gpu=0

./pix2fang_nest \
  --out_dir=./earth_10Mhz \
  --nside=4096 \
  --gpu=0

./pix2fang_nest \
  --out_dir=./earth_30Mhz \
  --nside=16384 \
  --gpu=0