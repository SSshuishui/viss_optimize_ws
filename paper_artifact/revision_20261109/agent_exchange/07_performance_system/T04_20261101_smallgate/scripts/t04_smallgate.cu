#include "orbit_online_mirror_tile_vissfast.cuh"
#include "viss_recon_kernel_mirror_tile_vissfast.cuh"
#include <complex>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

// Controlled synthetic fixture, not a new scientific dataset. Every device
// forward kernel below is included unchanged from the frozen T04 header.
constexpr int TP = 256;
constexpr int NH = 28;
constexpr int NF = 56;

struct DeviceFloats {
  std::vector<float*> owned;
  float* copy(const std::vector<float>& values) {
    float* pointer=nullptr;
    CHECK_CUDA(cudaMalloc(&pointer, values.size()*sizeof(float)));
    CHECK_CUDA(cudaMemcpy(pointer, values.data(), values.size()*sizeof(float), cudaMemcpyHostToDevice));
    owned.push_back(pointer);
    return pointer;
  }
  ~DeviceFloats() { for(float* pointer: owned) cudaFree(pointer); }
};

int main(int argc,char** argv) {
  int shared_arrays=5;
  std::string output="smallgate.csv";
  for(int i=1;i<argc;++i) {
    std::string arg=argv[i];
    if(arg.rfind("--shared-arrays=",0)==0) shared_arrays=std::stoi(arg.substr(16));
    else if(arg.rfind("--output=",0)==0) output=arg.substr(9);
    else return 3;
  }
  if(shared_arrays!=4 && shared_arrays!=5) return 3;
  CHECK_CUDA(cudaSetDevice(0));
  std::vector<float> B(TP),l(TP),m(TP),n(TP),nm1(TP);
  for(int p=0;p<TP;++p) {
    const double theta=(4.0+21.0*(p+0.5)/TP)*M_PI/180.;
    const double phi=p*2.399963229728653;
    l[p]=float(std::sin(theta)*std::cos(phi));
    m[p]=float(std::sin(theta)*std::sin(phi));
    n[p]=float(std::cos(theta)); nm1[p]=n[p]-1.0f;
    B[p]=float(1.+0.3*std::cos(p*.17)+0.2*std::sin(p*.31));
  }
  std::vector<float> u(NF),v(NF),w(NF),x1(NF),y1(NF),z1(NF),in1(NF,1.f),x2(NF),y2(NF),z2(NF),in2(NF,1.f);
  for(int j=0;j<NH;++j) {
    const float scale=(j%7==0)?0.0001f:((j%7==1)?0.05f:((j%7==2)?1.f:150.f*(j%7-1)));
    u[j]=scale*(0.23f+0.08f*j); v[j]=scale*(-0.51f+0.03f*j); w[j]=scale*(0.77f-0.01f*j);
    x1[j]=0.f; y1[j]=0.f; z1[j]=1.f;
    x2[j]=0.f; y2[j]=0.f; z2[j]=1.f;
    if(j/7==1 || j/7==3) { x2[j]=float(std::sqrt(.5)); z2[j]=-float(std::sqrt(.5)); }
    if(j/7==2) z2[j]=-1.f;
    if(j/7==3) { x1[j]=-float(std::sqrt(.5)); z1[j]=-float(std::sqrt(.5)); }
    u[j+NH]=-u[j]; v[j+NH]=-v[j]; w[j+NH]=-w[j];
    x1[j+NH]=x2[j]; y1[j+NH]=y2[j]; z1[j+NH]=z2[j];
    x2[j+NH]=x1[j]; y2[j+NH]=y1[j]; z2[j+NH]=z1[j];
  }
  DeviceFloats arrays;
  auto dB=arrays.copy(B),dl=arrays.copy(l),dm=arrays.copy(m),dn=arrays.copy(n),dnm1=arrays.copy(nm1);
  auto du=arrays.copy(u),dv=arrays.copy(v),dw=arrays.copy(w);
  auto dx1=arrays.copy(x1),dy1=arrays.copy(y1),dz1=arrays.copy(z1),di1=arrays.copy(in1);
  auto dx2=arrays.copy(x2),dy2=arrays.copy(y2),dz2=arrays.copy(z2),di2=arrays.copy(in2);
  auto dcx=arrays.copy({0.f}),dcy=arrays.copy({0.f}),dcz=arrays.copy({1.f});
  auto dcA=arrays.copy({float(std::cos(25.01*M_PI/180.))});
  auto dsA=arrays.copy({float(std::sin(25.01*M_PI/180.))});
  float2* dfull=nullptr; float2* dhalf=nullptr;
  CHECK_CUDA(cudaMalloc(&dfull,NF*sizeof(float2)));
  CHECK_CUDA(cudaMalloc(&dhalf,NH*sizeof(float2)));
  const float cosphi=-0.522349f;
  const size_t shared_bytes=shared_arrays*TP*sizeof(float);
  viss_partial_all_tilecone<TP,true><<<1,256,shared_bytes>>>(
    dB,dl,dm,dn,dnm1,TP,dcx,dcy,dcz,dcA,dsA,1,du,dv,dw,
    dx1,dy1,dz1,di1,dx2,dy2,dz2,di2,NF,cosphi,dfull);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  viss_partial_all_halfsym_tilecone<TP,true><<<1,256,5*TP*sizeof(float)>>>(
    dB,dl,dm,dn,dnm1,TP,dcx,dcy,dcz,dcA,dsA,1,du,dv,dw,
    dx1,dy1,dz1,di1,dx2,dy2,dz2,di2,NH,cosphi,dhalf);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  std::vector<float2> full(NF),half(NH);
  CHECK_CUDA(cudaMemcpy(full.data(),dfull,NF*sizeof(float2),cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(half.data(),dhalf,NH*sizeof(float2),cudaMemcpyDeviceToHost));
  std::ofstream file(output); file<<std::setprecision(17);
  file<<"baseline,mask_family,visible_pixels,minimum_point_mask_margin,max_abs_phase_rad,full_re,full_im,half_or_conjugate_re,half_or_conjugate_im,fp64_math_re,fp64_math_im,fp64_fixed_float_phase_re,fp64_fixed_float_phase_im\n";
  const float k=-2.f*(float)M_PI;
  for(int j=0;j<NF;++j) {
    std::complex<double> oracle(0.,0.),same_phase(0.,0.);
    int visible=0; double min_margin=1.e100,max_phase=0.;
    const float ku=k*u[j],kv=k*v[j],kw=k*w[j];
    for(int p=0;p<TP;++p) {
      const double c1=(double(l[p])*x1[j]+double(m[p])*y1[j]+double(n[p])*z1[j])*in1[j];
      const double c2=(double(l[p])*x2[j]+double(m[p])*y2[j]+double(n[p])*z2[j])*in2[j];
      min_margin=std::min(min_margin,std::min(std::abs(c1-cosphi),std::abs(c2-cosphi)));
      const double phase=-2.*M_PI*(double(u[j])*l[p]+double(v[j])*m[p]+double(w[j])*nm1[p]);
      max_phase=std::max(max_phase,std::abs(phase));
      if(c1<cosphi || c2<cosphi) continue;
      ++visible;
      oracle+=double(B[p])*std::complex<double>(std::cos(phase),std::sin(phase));
      const float rounded_phase=std::fma(kw,nm1[p],std::fma(kv,m[p],ku*l[p]));
      same_phase+=double(B[p])*std::complex<double>(std::cos(double(rounded_phase)),std::sin(double(rounded_phase)));
    }
    const int h=j%NH;
    file<<j<<','<<(h/7)<<','<<visible<<','<<min_margin<<','<<max_phase<<','
      <<full[j].x<<','<<full[j].y<<','<<half[h].x<<','<<((j<NH)?half[h].y:-half[h].y)<<','
      <<oracle.real()<<','<<oracle.imag()<<','<<same_phase.real()<<','<<same_phase.imag()<<'\n';
  }
  file.close();
  cudaFree(dfull); cudaFree(dhalf);
  std::cout<<"fixture=synthetic_1tile256_half28_full56 shared_arrays="<<shared_arrays<<" output="<<output<<'\n';
  return 0;
}
