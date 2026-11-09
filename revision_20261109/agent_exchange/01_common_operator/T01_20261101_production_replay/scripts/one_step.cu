// Actual production representative direction in a restricted-direction LS task.
// Reuse frozen replay helpers; rename its entry point, without editing it.
#define main frozen_replay_entry
#include "../source/replay.cu"
#undef main

template<typename T> static std::vector<T> read_vector(const std::string& path){
  std::ifstream in(path,std::ios::binary|std::ios::ate);auto size=in.tellg();
  if(size<0 || (size_t)size%sizeof(T))throw std::runtime_error("invalid vector "+path);
  in.seekg(0);std::vector<T> v((size_t)size/sizeof(T));in.read((char*)v.data(),size);
  if(!in.good())throw std::runtime_error("short vector "+path);return v;
}

template<int TILE>
__global__ void two_forward_directions(const ReplayRow* rows,int nb,
                    const float* l,const float* m,const float* n,int np,
                    const double* g,const double* d,float cosphi,double4* out){
  __shared__ float sl[TILE],sm[TILE],sn[TILE];
  __shared__ double sg[TILE],sd[TILE];
  int b=blockIdx.x*blockDim.x+threadIdx.x;bool active=b<nb;
  ReplayRow row{};if(active)row=rows[b];
  double gr=0,gi=0,dr=0,di=0;
  for(int p0=0;p0<np;p0+=TILE){
    for(int j=threadIdx.x;j<TILE;j+=blockDim.x){int p=p0+j;
      sl[j]=p<np?l[p]:0;sm[j]=p<np?m[p]:0;sn[j]=p<np?n[p]:0;
      sg[j]=p<np?g[p]:0;sd[j]=p<np?d[p]:0;}
    __syncthreads();
    if(active){for(int j=0;j<TILE && p0+j<np;j++){
      float ll=sl[j],mm=sm[j],nn=sn[j];
      float c1=(ll*row.x1+mm*row.y1+nn*row.z1)*row.inv1;
      float c2=(ll*row.x2+mm*row.y2+nn*row.z2)*row.inv2;
      if(c1<cosphi||c2<cosphi)continue;
      double phase=2*M_PI*((double)row.u*ll+(double)row.v*mm+(double)row.w*nn),s,c;
      sincos(phase,&s,&c);gr+=sg[j]*c;gi-=sg[j]*s;dr+=sd[j]*c;di-=sd[j]*s;
    }}__syncthreads();
  }
  if(active)out[b]=make_double4(gr,gi,dr,di);
}

int main(int argc,char** argv){try{
  std::string gpu=get_arg(argc,argv,"--gpus","3");setenv("CUDA_VISIBLE_DEVICES",gpu.c_str(),1);CHECK_CUDA(cudaSetDevice(0));
  std::string cache=get_arg(argc,argv,"--cache_root","");std::string input=get_arg(argc,argv,"--case_dir","");
  std::string output=get_arg(argc,argv,"--output","");int day=to_int(get_arg(argc,argv,"--day","1"),1);
  if(cache.empty()||input.empty()||output.empty())throw std::runtime_error("cache/case/output required");
  int nside=0;auto rows=read_day(cache,day,nside);
  int noise_seed=to_int(get_arg(argc,argv,"--noise_seed","0"),0);
  if(noise_seed){
    std::string noise_root=get_arg(argc,argv,"--noise_root","");std::string dcf=get_arg(argc,argv,"--dcf_bin","");
    if(noise_root.empty()||dcf.empty())throw std::runtime_error("noise_root/dcf_bin required for noise");
    ensure_dir(noise_root);ensure_dir(noise_root+"/cache");ensure_dir(noise_root+"/runs/day1_s10_g1_n4096");disk_guard(noise_root);
    long double energy=0;for(auto r:rows)energy+=(long double)r.re*r.re+(long double)r.im*r.im;
    double sigma=std::sqrt((double)(energy/rows.size()))*.1/std::sqrt(2.0);
    std::mt19937_64 rng((uint64_t)noise_seed);std::normal_distribution<double> normal(0.0,1.0);
    long double noise_energy=0;
    for(auto& r:rows){float original_re=r.re,original_im=r.im;
      r.re=(float)((double)r.re+sigma*normal(rng));r.im=(float)((double)r.im+sigma*normal(rng));
      noise_energy+=((long double)r.re-original_re)*((long double)r.re-original_re)+((long double)r.im-original_im)*((long double)r.im-original_im);
      float angle=-2.0f*(float)M_PI*r.w,c=std::cos(angle),s=std::sin(angle);
      r.raw_re=r.re*c+r.im*s;r.raw_im=r.im*c-r.re*s;}
    // Preserve half-set complex noise; negative baselines are implicit conjugates.
    int nt=(int)rows.size()/28,t0=0;
    for(int seg=1;seg<=10;seg++){
      int tlen=nt/10+(seg<=nt%10);ReplayHeader hdr{{'P','R','E','P','L','A','Y','1'},nside,day,seg,t0,tlen,tlen*28,1};
      std::string file=noise_root+"/cache/day"+std::to_string(day)+"_seg"+std::to_string(seg)+".bin";
      std::ofstream out(file,std::ios::binary);out.write((char*)&hdr,sizeof(hdr));out.write((char*)(rows.data()+(size_t)t0*28),(size_t)tlen*28*sizeof(ReplayRow));
      out.close();if(!out.good())throw std::runtime_error("noise cache write failed");t0+=tlen;
    }
    std::ofstream meta(noise_root+"/noise_metadata.json");meta<<std::setprecision(17)<<"{\"seed\":"<<noise_seed<<",\"snr_db\":20,\"realized_snr_db\":"<<(double)(10*log10l(energy/noise_energy))<<",\"per_component_sigma\":"<<sigma
      <<",\"generator\":\"C++ mt19937_64 + standard normal_distribution, seed fixes this recorded source/binary/platform\",\"stage\":\"independent complex Gaussian noise added to fullsky consumed-n half-visibilities after host phase correction; raw_nm1 fields reconstructed by inverse float host rotation solely for diagnostic consistency, not new forward simulation; negative half conjugate implied\"}\n";meta.close();
    input=noise_root+"/runs/day1_s10_g1_n4096";
    std::vector<std::string> av={"replay-noise","--gpus="+gpu,"--cache_root="+noise_root+"/cache","--out_dir="+input,"--day="+std::to_string(day),"--segs=10","--samples=4096","--grid_scale=1","--rotation=0","--dcf_bin="+dcf};
    std::vector<char*> argv2;for(auto& value:av)argv2.push_back(value.data());
    if(frozen_replay_entry((int)argv2.size(),argv2.data()))throw std::runtime_error("noisy production replay failed");
  }
  auto l=read_vector<float>(input+"/l.f32"),m=read_vector<float>(input+"/m.f32"),n=read_vector<float>(input+"/n.f32");
  auto g=read_vector<double>(input+"/direct_N.f64");auto repN=read_vector<float>(input+"/rep_N.f32");auto repW=read_vector<uint32_t>(input+"/rep_W.u32");
  size_t np=l.size();if(m.size()!=np||n.size()!=np||g.size()!=np||repN.size()!=np||repW.size()!=np)throw std::runtime_error("direction shape mismatch");
  std::vector<double> d(np);long double gg=0,gd=0,dd=0;
  for(size_t p=0;p<np;p++){d[p]=repW[p]>0?(double)repN[p]/repW[p]:0;gg+=(long double)g[p]*g[p];gd+=(long double)g[p]*d[p];dd+=(long double)d[p]*d[p];}
  auto dl=device_copy(l),dm=device_copy(m),dn=device_copy(n);auto dg=device_copy(g),ddir=device_copy(d);auto dr=device_copy(rows);
  double4* applied=nullptr;CHECK_CUDA(cudaMalloc(&applied,rows.size()*sizeof(double4)));
  float R=1737.1e3f,h=300e3f;float cosphi=cosf((float)M_PI-asinf(R/(R+h)));
  HostTimer timer;timer.tic();
  two_forward_directions<256><<<((int)rows.size()+127)/128,128>>>(dr,(int)rows.size(),dl,dm,dn,(int)np,dg,ddir,cosphi,applied);
  CHECK_CUDA(cudaPeekAtLastError());auto a=host_copy(applied,rows.size());double compute_s=timer.toc_s();
  long double y2=0,ag2=0,ad2=0,agy=0,ady=0;
  for(size_t b=0;b<rows.size();b++){
    auto y=rows[b];auto z=a[b];y2+=(long double)y.re*y.re+(long double)y.im*y.im;
    ag2+=(long double)z.x*z.x+(long double)z.y*z.y;ad2+=(long double)z.z*z.z+(long double)z.w*z.w;
    agy+=(long double)z.x*y.re+(long double)z.y*y.im;ady+=(long double)z.z*y.re+(long double)z.w*y.im;
  }
  long double alpha_g=agy/ag2,alpha_d=ady/ad2;
  long double decrease_g=.5L*agy*agy/ag2,decrease_d=.5L*ady*ady/ad2;
  long double dot_g_error=fabsl(agy-gg)/std::max(fabsl(gg),1e-300L),dot_d_error=fabsl(ady-gd)/std::max(fabsl(gd),1e-300L);
  bool pass=dot_g_error<1e-8L && dot_d_error<1e-8L && ag2>0 && ad2>0;
  std::ofstream out(output);out<<std::setprecision(17)
    <<"{\n\"day\":"<<day<<",\"sample_directions\":"<<np<<",\"full_half_baselines\":"<<rows.size()<<","
    <<"\"objective\":\"f(x)=0.5 sum_half |A_subset x-y_fullsky|^2, real unconstrained x, x0=0; A uses production frozen mask and consumed n exact phase, no extra pixel-area factor in unknown coefficients\","
    <<"\"g_definition\":\"+Re(A_subset^H y), raw direct N, negative of true gradient at0; no N/W normalization\","
    <<"\"d_definition\":\"actual original CUDA representative raw N / occupied-visible-group W; W0 is zero output\","
    <<"\"g_norm_squared\":"<<(double)gg<<",\"d_norm_squared\":"<<(double)dd<<",\"g_dot_d\":"<<(double)gd<<","
    <<"\"direction_cosine\":"<<(double)(gd/sqrtl(gg*dd))<<",\"directional_derivative_along_positive_d\":"<<(double)-gd<<","
    <<"\"A_g_norm_squared\":"<<(double)ag2<<",\"A_d_norm_squared\":"<<(double)ad2<<","
    <<"\"initial_loss\":"<<(double)(y2*.5L)<<",\"optimal_line_alpha_g\":"<<(double)alpha_g<<",\"optimal_line_alpha_d\":"<<(double)alpha_d<<","
    <<"\"line_loss_decrease_g\":"<<(double)decrease_g<<",\"line_loss_decrease_d\":"<<(double)decrease_d<<","
    <<"\"decrease_d_over_g\":"<<(double)(decrease_d/decrease_g)<<","
    <<"\"g_decrease_fraction_of_initial\":"<<(double)(decrease_g/(.5L*y2))<<",\"d_decrease_fraction_of_initial\":"<<(double)(decrease_d/(.5L*y2))<<","
    <<"\"Ag_y_vs_g_g_relative_error\":"<<(double)dot_g_error<<",\"Ad_y_vs_g_d_relative_error\":"<<(double)dot_d_error<<","
    <<"\"adjoint_scalar_controls_pass\":"<<(pass?"true":"false")<<",\"kernel_plus_D2H_wall_s\":"<<compute_s<<","
    <<"\"noise_seed\":"<<noise_seed<<",\"noise_snr_db\":"<<(noise_seed?20:0)<<","
    <<"\"noise\":\""<<(noise_seed?"one independent20dB complex Gaussian half-vis realization at consumed-n stage":"none; cached noiseless fullsky production visibility")<<"\","
    <<"\"limits\":\"restricted sampled-direction fit to fullsky visibility, not a sky-recovery problem, not positivity constrained, one exact line-search update does not prove convergence; shared-node timing only\"}\n";
  out.close();if(!out.good())throw std::runtime_error("result write failed");
  std::cout<<"ONE_STEP_DONE pass="<<pass<<" cosine="<<(double)(gd/sqrtl(gg*dd))<<" decrease_ratio="<<(double)(decrease_d/decrease_g)<<" wall="<<compute_s<<"\n";
  return pass?0:3;
}catch(const std::exception& e){std::cerr<<"ERROR "<<e.what()<<"\n";return 2;}}
