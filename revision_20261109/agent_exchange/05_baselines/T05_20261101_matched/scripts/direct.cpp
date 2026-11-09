#include <cmath>
#include <cstdint>
#include <omp.h>
extern "C" void apply_direct(int ns,int nt,const double* s,const double* t,const double* c,const uint8_t* mask,int masked,int sign,int threads,double* out){
  const double tau=6.283185307179586476925286766559;
  #pragma omp parallel for num_threads(threads) schedule(static)
  for(int p=0;p<nt;p++){
    double ar=0,ai=0; const double x=t[3*p],y=t[3*p+1],z=t[3*p+2];
    for(int j=0;j<ns;j++){
      if(masked && !mask[(size_t)j*nt+p])continue;
      const double ph=sign*tau*((s[3*j]*x+s[3*j+1]*y)+s[3*j+2]*z);
      double sn,cs;sincos(ph,&sn,&cs);
      ar+=c[2*j]*cs-c[2*j+1]*sn;ai+=c[2*j]*sn+c[2*j+1]*cs;
    }
    out[2*p]=ar;out[2*p+1]=ai;
  }
}
