#pragma once
#include "raw_stream_moments.hpp"
// Independent correctness/export helper. Never included by formal timing source.
__global__ void diag_gather_W(const double* w,const uint32_t* ids,int count,double* out){int p=blockIdx.x*blockDim.x+threadIdx.x;if(p<count)out[p]=w[ids[p]];}
__global__ void diag_normalize_C(float* n,const double* w,long long count){long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(p<count)n[p]=(w[p]>0)?(float)((double)n[p]/w[p]):0.0f;}

static std::string diag_scan_weight(const double* W,long long np,long long* zero_count){
 const size_t CH=1<<20;std::vector<double>part(CH);SHA256_CTX sh;SHA256_Init(&sh);long long finite=0,zeros=0;double lo=INFINITY,hi=-INFINITY;
 for(long long off=0;off<np;off+=CH){size_t take=(size_t)std::min<long long>(CH,np-off);CHECK_CUDA(cudaMemcpy(part.data(),W+off,take*sizeof(double),cudaMemcpyDeviceToHost));SHA256_Update(&sh,part.data(),take*sizeof(double));for(size_t j=0;j<take;j++){double v=part[j];if(!std::isfinite(v)||v<0)throw std::runtime_error("invalid full W");finite++;zeros+=(v==0);lo=std::min(lo,v);hi=std::max(hi,v);}}
 if(zero_count)*zero_count=zeros;unsigned char digest[SHA256_DIGEST_LENGTH];SHA256_Final(digest,&sh);std::ostringstream o;o<<std::setprecision(17)<<"{\"dtype\":\"float64\",\"count\":"<<np<<",\"finite_count\":"<<finite<<",\"zero_count\":"<<zeros<<",\"min\":"<<lo<<",\"max\":"<<hi<<",\"sha256\":\""<<hex_sha(digest)<<"\"}";return o.str();
}

static void diag_compare_original_rawN(const float* N,const double* W,const float* direction_n,long long np,long long zero_weights,const std::string& grid_file,const std::string& actual_file,const std::string& out){
 std::ifstream grid(grid_file,std::ios::binary|std::ios::ate),actual(actual_file,std::ios::binary|std::ios::ate);
 if(!grid.good()||!actual.good()||grid.tellg()!=(std::streamoff)(np*sizeof(float))||actual.tellg()!=(std::streamoff)(np*sizeof(float)))throw std::runtime_error("raw metrics require two prevalidated complete STS C files");grid.seekg(0);actual.seekg(0);
 const size_t CH=1<<20;std::vector<float>x(CH),y0(CH),y1(CH),n(CH);std::vector<double>w(zero_weights?CH:0);std::vector<uint8_t>tags(CH);
 RawPairMoments moments[2][2][4];long long bands[3]={},valid_bands[3]={},observed_zero=0;const double high=std::sqrt(3.0)/2.0;
 for(long long off=0;off<np;off+=CH){size_t take=(size_t)std::min<long long>(CH,np-off);CHECK_CUDA(cudaMemcpy(x.data(),N+off,take*sizeof(float),cudaMemcpyDeviceToHost));CHECK_CUDA(cudaMemcpy(n.data(),direction_n+off,take*sizeof(float),cudaMemcpyDeviceToHost));if(zero_weights)CHECK_CUDA(cudaMemcpy(w.data(),W+off,take*sizeof(double),cudaMemcpyDeviceToHost));grid.read((char*)y0.data(),take*sizeof(float));actual.read((char*)y1.data(),take*sizeof(float));if(!grid.good()||!actual.good())throw std::runtime_error("raw metrics STS short read");
  for(size_t j=0;j<take;j++){if(!std::isfinite(n[j]))throw std::runtime_error("raw metric nonfinite direction");double a=std::fabs((double)n[j]);uint8_t band=a<=.5?0:(a<=high?1:2);tags[j]=band;bands[band]++;bool valid=!zero_weights||w[j]>0;if(valid)valid_bands[band]++;else observed_zero++;}
  const float* ys[2]={y0.data(),y1.data()};for(int method=0;method<2;method++)for(int band=-1;band<3;band++){moments[method][0][band+1].add_block(x.data(),ys[method],tags.data(),nullptr,take,band,false);if(zero_weights)moments[method][1][band+1].add_block(x.data(),ys[method],tags.data(),w.data(),take,band,true);}
 }
 if(grid.peek()!=std::char_traits<char>::eof()||actual.peek()!=std::char_traits<char>::eof()||observed_zero!=zero_weights)throw std::runtime_error("raw metrics EOF/Wzero mismatch");if(!zero_weights)for(int method=0;method<2;method++)for(int band=0;band<4;band++)moments[method][1][band]=moments[method][0][band];
 std::pair<long double,long double> fits[2][2];long double direct_sse[2][2][4]={};for(int method=0;method<2;method++)for(int domain=0;domain<2;domain++)fits[method][domain]=moments[method][domain][0].affine();
 // Evaluate fitted residuals directly in a second read-only stream. This avoids
 // subtracting enormous covariance terms when shared zeros make rho nearly1.
 grid.clear();actual.clear();grid.seekg(0);actual.seekg(0);
 for(long long off=0;off<np;off+=CH){size_t take=(size_t)std::min<long long>(CH,np-off);CHECK_CUDA(cudaMemcpy(x.data(),N+off,take*sizeof(float),cudaMemcpyDeviceToHost));CHECK_CUDA(cudaMemcpy(n.data(),direction_n+off,take*sizeof(float),cudaMemcpyDeviceToHost));if(zero_weights)CHECK_CUDA(cudaMemcpy(w.data(),W+off,take*sizeof(double),cudaMemcpyDeviceToHost));grid.read((char*)y0.data(),take*sizeof(float));actual.read((char*)y1.data(),take*sizeof(float));if(!grid.good()||!actual.good())throw std::runtime_error("affine residual stream short read");
  for(size_t j=0;j<take;j++){double a=std::fabs((double)n[j]);int band=a<=.5?0:(a<=high?1:2);const float ys[2]={y0[j],y1[j]};for(int method=0;method<2;method++){auto f=fits[method][0];long double e=f.first*(long double)ys[method]+f.second-(long double)x[j];direct_sse[method][0][0]+=e*e;direct_sse[method][0][band+1]+=e*e;if(zero_weights&&w[j]>0){f=fits[method][1];e=f.first*(long double)ys[method]+f.second-(long double)x[j];direct_sse[method][1][0]+=e*e;direct_sse[method][1][band+1]+=e*e;}}}
 }
 if(grid.peek()!=std::char_traits<char>::eof()||actual.peek()!=std::char_traits<char>::eof())throw std::runtime_error("affine residual stream EOF");if(!zero_weights)for(int method=0;method<2;method++)for(int band=0;band<4;band++)direct_sse[method][1][band]=direct_sse[method][0][band];
 const char* names[2]={"original_STS_grid_C","actual_uv_STS_C"};const char* domains[2]={"all201M_including_W0","secondary_reference_W_gt0"};const char* regions[4]={"full_domain","native_abs_lat_le30","native_abs_lat_30to60","native_abs_lat_gt60"};
 std::ofstream f(out+"/raw_output_full_metrics.json");f<<"{\"reference_identity\":\"original scientific raw DCF weighted sum N, corrected index, no pixel normalization\",\"candidate_identity\":\"STS normalized C=Ncell/Wcell\",\"uncalibrated_L2_note\":\"different accumulation/normalization units; recorded amplitude difference, not solely approximation error\",\"reference_W0_count\":"<<zero_weights<<",\"all_native_band_counts\":["<<bands[0]<<","<<bands[1]<<","<<bands[2]<<"],\"valid_native_band_counts\":["<<valid_bands[0]<<","<<valid_bands[1]<<","<<valid_bands[2]<<"],\"metrics\":{";
 for(int method=0;method<2;method++){if(method)f<<",";f<<"\""<<names[method]<<"\":{";for(int domain=0;domain<2;domain++){if(domain)f<<",";auto fit=fits[method][domain];f<<"\""<<domains[domain]<<"\":{\"global_alpha\":"<<std::setprecision(20)<<(double)fit.first<<",\"global_beta\":"<<(double)fit.second<<",\"global_fit_scope\":\"candidate-to-reference, one fit over this full domain; reused unchanged for its native regions\",\"regions\":{";for(int band=0;band<4;band++){if(band)f<<",";f<<"\""<<regions[band]<<"\":"<<moments[method][domain][band].json(fit.first,fit.second,&direct_sse[method][domain][band]);}f<<"}}";}f<<"}";}f<<"},\"native_frame\":\"actual GPU n, not Galactic\",\"algorithm\":\"centered-block Chan fit/rho; fitted residuals accumulated directly in second read-only stream, long double\",\"rawN_or_fullW_written\":false}\n";
}

static void diag_finalize_export(float* N,const double* W,const float* direction_n,long long np,const std::string& out,const std::string& rawN_json,const std::string& grid_C,const std::string& actual_C){
 long long zero_count=0;const std::string weight_json=diag_scan_weight(W,np,&zero_count);
 diag_compare_original_rawN(N,W,direction_n,np,zero_count,grid_C,actual_C,out);
 diag_normalize_C<<<(np+255)/256,256>>>(N,W,np);CHECK_CUDA(cudaDeviceSynchronize());
 const size_t CH=1<<20;std::vector<float>c(CH),n(CH);std::vector<double>w(CH);std::vector<uint8_t>tags(CH);
 std::ofstream image(out+"/corrected_dcf_C.f32",std::ios::binary),regions(out+"/native_regions.u8",std::ios::binary);if(!image.good()||!regions.good())throw std::runtime_error("temporary C/region open");
 SHA256_CTX cs,ts;SHA256_Init(&cs);SHA256_Init(&ts);double lo=INFINITY,hi=-INFINITY;long long finite=0,population[4]={};
 const double high_boundary=std::sqrt(3.0)/2.0;
 for(long long off=0;off<np;off+=CH){size_t take=(size_t)std::min<long long>(CH,np-off);CHECK_CUDA(cudaMemcpy(c.data(),N+off,take*sizeof(float),cudaMemcpyDeviceToHost));CHECK_CUDA(cudaMemcpy(n.data(),direction_n+off,take*sizeof(float),cudaMemcpyDeviceToHost));CHECK_CUDA(cudaMemcpy(w.data(),W+off,take*sizeof(double),cudaMemcpyDeviceToHost));
  for(size_t j=0;j<take;j++){if(!std::isfinite(c[j])||!std::isfinite(n[j])||!std::isfinite(w[j]))throw std::runtime_error("nonfinite C/region");finite++;lo=std::min(lo,(double)c[j]);hi=std::max(hi,(double)c[j]);double a=std::fabs((double)n[j]);uint8_t tag=(w[j]<=0)?255:(a<=.5?0:(a<=high_boundary?1:2));tags[j]=tag;population[tag==255?3:tag]++;}
  image.write((char*)c.data(),take*sizeof(float));regions.write((char*)tags.data(),take);if(!image.good()||!regions.good())throw std::runtime_error("temporary C/region write");SHA256_Update(&cs,c.data(),take*sizeof(float));SHA256_Update(&ts,tags.data(),take);
 }
 image.close();regions.close();unsigned char cd[SHA256_DIGEST_LENGTH],td[SHA256_DIGEST_LENGTH];SHA256_Final(cd,&cs);SHA256_Final(td,&ts);
 std::ofstream report(out+"/full_stats.json");report<<std::setprecision(17)<<"{\"ordering\":\"NEST\",\"rawN\":"<<rawN_json<<",\"W\":"<<weight_json<<",\"C\":{\"dtype\":\"float32\",\"count\":"<<np<<",\"finite_count\":"<<finite<<",\"min\":"<<lo<<",\"max\":"<<hi<<",\"sha256\":\""<<hex_sha(cd)<<"\",\"temporary_file\":\"corrected_dcf_C.f32\"},\"native_regions\":{\"dtype\":\"uint8\",\"bytes\":"<<np<<",\"sha256\":\""<<hex_sha(td)<<"\",\"counts\":["<<population[0]<<","<<population[1]<<","<<population[2]<<","<<population[3]<<"],\"labels\":\"0:abs(GPU_n)<=.5;1:.5<abs(GPU_n)<=sqrt(3)/2;2:higher;255:W0;native frame, not Galactic\"},\"normalization\":\"diagnostic float32(rawN/float64W), rawN math retained\",\"temporary_C_written\":true,\"full_W_written\":false}\n";
}
