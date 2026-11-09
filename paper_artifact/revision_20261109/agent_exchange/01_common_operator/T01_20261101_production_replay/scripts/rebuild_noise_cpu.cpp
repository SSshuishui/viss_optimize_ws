// CPU-only deterministic reconstruction of the recorded consumed-n noise cache.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <filesystem>
struct Row{float u,v,w,x1,y1,z1,inv1,x2,y2,z2,inv2,raw_re,raw_im,re,im;};
struct Header{char magic[8];int32_t nside,day,seg,t0,tlen,nhalf,gpus;};
static_assert(sizeof(Row)==60 && sizeof(Header)==36,"layout");
int main(int argc,char** argv){
  if(argc!=3)return 2;std::string original=argv[1],dest=argv[2];std::vector<Row> rows;std::vector<Header> headers;
  for(int s=1;s<=10;s++){std::ifstream in(original+"/day1_seg"+std::to_string(s)+".bin",std::ios::binary);Header h{};in.read((char*)&h,sizeof(h));if(!in.good()||h.nhalf<=0)return 3;headers.push_back(h);size_t at=rows.size();rows.resize(at+h.nhalf);in.read((char*)(rows.data()+at),(size_t)h.nhalf*sizeof(Row));if(!in.good())return 4;}
  long double energy=0;for(auto r:rows)energy+=(long double)r.re*r.re+(long double)r.im*r.im;
  double sigma=std::sqrt((double)(energy/rows.size()))*.1/std::sqrt(2.0);std::mt19937_64 rng(101);std::normal_distribution<double> normal(0,1);
  for(auto& r:rows){r.re=(float)((double)r.re+sigma*normal(rng));r.im=(float)((double)r.im+sigma*normal(rng));float angle=-2.0f*(float)3.14159265358979323846*r.w,c=std::cos(angle),s=std::sin(angle);r.raw_re=r.re*c+r.im*s;r.raw_im=r.im*c-r.re*s;}
  std::filesystem::create_directories(dest);size_t at=0;
  for(auto h:headers){std::ofstream out(dest+"/day1_seg"+std::to_string(h.seg)+".bin",std::ios::binary);out.write((char*)&h,sizeof(h));out.write((char*)(rows.data()+at),(size_t)h.nhalf*sizeof(Row));if(!out.good())return 5;at+=h.nhalf;}
  std::cout<<"CPU_NOISE_REBUILT rows="<<rows.size()<<" sigma="<<sigma<<"\n";return 0;
}
