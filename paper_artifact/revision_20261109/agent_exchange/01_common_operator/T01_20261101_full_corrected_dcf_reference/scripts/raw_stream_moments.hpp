#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <utility>
#include <iomanip>
#include <sstream>
#include <stdexcept>

struct RawPairMoments {
 uint64_t count=0;long double mx=0,my=0,xx=0,yy=0,xy=0,squared_error=0;
 long double low=INFINITY,high=-INFINITY,max_error=0;
 void merge(const RawPairMoments& b){
  if(!b.count)return;if(!count){*this=b;return;}
  const uint64_t total=count+b.count;long double dx=b.mx-mx,dy=b.my-my,k=(long double)count*b.count/total;
  xx+=b.xx+dx*dx*k;yy+=b.yy+dy*dy*k;xy+=b.xy+dx*dy*k;mx+=dx*b.count/total;my+=dy*b.count/total;count=total;
  squared_error+=b.squared_error;low=std::min(low,b.low);high=std::max(high,b.high);max_error=std::max(max_error,b.max_error);
 }
 void add_block(const float* x,const float* y,const uint8_t* native_band,const double* W,std::size_t size,int band,bool valid_only){
  if(valid_only&&!W)throw std::runtime_error("valid domain requires W block");
  RawPairMoments b;long double sx=0,sy=0;
  for(std::size_t i=0;i<size;i++){
   if((band>=0&&native_band[i]!=(uint8_t)band)||(valid_only&&W[i]<=0))continue;
   long double a=x[i],c=y[i];if(!std::isfinite(a)||!std::isfinite(c))throw std::runtime_error("nonfinite raw-output metrics input");
   b.count++;sx+=a;sy+=c;b.low=std::min(b.low,a);b.high=std::max(b.high,a);long double d=c-a;b.squared_error+=d*d;b.max_error=std::max(b.max_error,std::fabs(d));
  }
  if(!b.count)return;b.mx=sx/b.count;b.my=sy/b.count;
  for(std::size_t i=0;i<size;i++){
   if((band>=0&&native_band[i]!=(uint8_t)band)||(valid_only&&W[i]<=0))continue;
   long double dx=(long double)x[i]-b.mx,dy=(long double)y[i]-b.my;b.xx+=dx*dx;b.yy+=dy*dy;b.xy+=dx*dy;
  }
  merge(b);
 }
 std::pair<long double,long double> affine()const{
  long double alpha=yy>0?xy/yy:0;return {alpha,mx-alpha*my};
 }
 std::string json(long double alpha,long double beta,const long double* directly_accumulated_affine_sse=nullptr)const{
  if(!count)return "{\"count\":0}";
  const long double norm=xx+count*mx*mx,reference_std=std::sqrt(xx/count),rmse=std::sqrt(squared_error/count),mean_error=alpha*my+beta-mx;
  const long double calibrated_sse=directly_accumulated_affine_sse?*directly_accumulated_affine_sse:std::max((long double)0,alpha*alpha*yy+xx-2*alpha*xy+count*mean_error*mean_error),calibrated_rmse=std::sqrt(calibrated_sse/count);
  std::ostringstream s;s<<std::setprecision(20)<<"{\"count\":"<<count<<",\"reference_mean\":"<<(double)mx<<",\"candidate_mean\":"<<(double)my<<",\"reference_std\":"<<(double)reference_std<<",\"uncalibrated_RMSE\":"<<(double)rmse<<",\"uncalibrated_relative_L2\":";
  if(norm>0)s<<(double)std::sqrt(squared_error/norm);else s<<"null";
  s<<",\"uncalibrated_RMSE_over_reference_std\":";if(reference_std>0)s<<(double)(rmse/reference_std);else s<<"null";
  s<<",\"pearson\":";if(xx>0&&yy>0)s<<(double)(xy/std::sqrt(xx*yy));else s<<"null";
  s<<",\"max_abs_uncalibrated_error\":"<<(double)max_error<<",\"same_global_affine_after\":{\"alpha\":"<<(double)alpha<<",\"beta\":"<<(double)beta<<",\"RMSE\":"<<(double)calibrated_rmse<<",\"RMSE_over_reference_std\":";
  if(reference_std>0)s<<(double)(calibrated_rmse/reference_std);else s<<"null";
  s<<",\"relative_L2\":";if(norm>0)s<<(double)std::sqrt(calibrated_sse/norm);else s<<"null";s<<"}}";return s.str();
 }
};
