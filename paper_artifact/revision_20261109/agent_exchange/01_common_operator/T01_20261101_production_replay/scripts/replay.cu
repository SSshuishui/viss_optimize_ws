#include "common_mirror_tile_vissfast.hpp"
#include "orbit_online_mirror_tile_vissfast.cuh"
#include "viss_recon_kernel_mirror_tile_vissfast.cuh"
#include "replay_common.hpp"

struct LegacyDcfHeader {char magic[8];uint32_t version;char btag[8];float lambda_m,bl_max;int32_t days;uint64_t seed,mb_len,dcf_len;};
static std::vector<float> load_legacy_dcf(const std::string& path){
  std::ifstream in(path,std::ios::binary);LegacyDcfHeader h{};in.read((char*)&h,sizeof(h));
  if(!in.good() || std::memcmp(h.magic,"DCFMBIN1",8) || h.version!=1 || h.dcf_len>1000000)
    throw std::runtime_error("invalid frozen legacy DCF header");
  in.seekg(h.mb_len*sizeof(int64_t),std::ios::cur);std::vector<float> d(h.dcf_len);
  in.read((char*)d.data(),d.size()*sizeof(float));if(!in.good())throw std::runtime_error("short DCF read");
  std::cout<<"LEGACY_DCF btag="<<h.btag<<" lambda="<<h.lambda_m<<" days="<<h.days<<" seed="<<h.seed<<" n="<<d.size()<<"\n";
  return d;
}
__global__ void legacy_weights(const ReplayRow* rows,int n,const float* dcf,int length,float* out){
  int h=blockIdx.x*blockDim.x+threadIdx.x;if(h>=n)return;auto r=rows[h];
  float bl=sqrtf(r.u*r.u+r.v*r.v+r.w*r.w),wgt=0;
  if(length>0){if(!isfinite(bl)||bl<=1e-20f)wgt=dcf[0];else{
    int gs=(int)ceilf((bl-0.25f)/0.5f)+1;gs=max(0,min(length-1,gs));
    float s=fminf(1.0f,fmaxf(-1.0f,r.w/bl));float c=sqrtf(fmaxf(0.0f,1-s*s));
    float gdg=c>1e-12f?1.5f*sqrtf(fabsf(1-4*s*s))/c:0;wgt=dcf[gs]*gdg;
  }}
  if(!isfinite(wgt)||wgt<0)wgt=0;out[h]=2.0f*fminf(0.125f,wgt);
}
// Each block partitions the baseline reduction; all phase/reduction math below
// uses binary64, whereas the point mask deliberately reuses production fp32.
__global__ void control_partial(const ReplayRow* rows,const float* weight,int nrows,
                                const float* l,const float* m,const float* n,int np,
                                const float* gu,const float* gv,float fa,float fb,float cosphi,
                                int part_size,int parts,double* out){
  int p=blockIdx.x*blockDim.x+threadIdx.x,part=blockIdx.y;if(p>=np)return;
  double a[9]={};float ll=l[p],mm=m[p],nn=n[p];
  int hi=min(nrows,(part+1)*part_size);
  for(int h=part*part_size;h<hi;h++){
    ReplayRow r=rows[h];float c1=(ll*r.x1+mm*r.y1+nn*r.z1)*r.inv1;
    float c2=(ll*r.x2+mm*r.y2+nn*r.z2)*r.inv2;
    if(c1<cosphi || c2<cosphi)continue;
    double lp=(double)ll+(double)fa*nn,mp=(double)mm+(double)fb*nn;
    double pd=2.0*M_PI*((double)r.u*ll+(double)r.v*mm+(double)r.w*nn);
    double pp=2.0*M_PI*((double)r.u*lp+(double)r.v*mp);
    double pq=2.0*M_PI*((double)gu[h]*lp+(double)gv[h]*mp);
    float nm1=nn-1.0f;
    double pr=2.0*M_PI*((double)r.u*ll+(double)r.v*mm+(double)r.w*nm1);
    double s,c;sincos(pd,&s,&c);double vd=(double)r.re*c-(double)r.im*s;
    sincos(pp,&s,&c);double vp=(double)r.re*c-(double)r.im*s;
    sincos(pq,&s,&c);double vq=(double)r.re*c-(double)r.im*s;
    sincos(pr,&s,&c);double vr=(double)r.raw_re*c-(double)r.raw_im*s;
    a[0]+=vd;a[1]+=vp;a[2]+=vq;a[3]+=vr;
    a[4]+=weight[h]*vd;a[5]+=weight[h]*vp;a[6]+=weight[h]*vq;
    a[7]+=1;a[8]+=weight[h];
  }
  for(int k=0;k<9;k++)out[((size_t)k*parts+part)*np+p]=a[k];
}

// Exact-3D and plane controls with exactly the visible-member group averaging
// and per-pixel group denominator used by production occ_mode=0. These are a
// second, explicitly cell-weighted operator, separate from natural sampling.
__global__ void cell_controls(const ReplayRow* rows,const int* offsets,const int* idx,
                             const float* gu,const float* gv,int ngroups,
                             const float* l,const float* m,const float* n,int np,
                             float fa,float fb,float cosphi,double* out){
  int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=np)return;
  float ll=l[p],mm=m[p],nn=n[p];double N[4]={};
  double lp=(double)ll+(double)fa*nn,mp=(double)mm+(double)fb*nn;
  for(int q=0;q<ngroups;q++){
    double vd=0,vp=0,vr=0,vi=0;int count=0;
    for(int pos=offsets[q];pos<offsets[q+1];pos++){
      ReplayRow r=rows[idx[pos]];
      float c1=(ll*r.x1+mm*r.y1+nn*r.z1)*r.inv1,c2=(ll*r.x2+mm*r.y2+nn*r.z2)*r.inv2;
      if(c1<cosphi||c2<cosphi)continue;
      double s,c;sincos(2*M_PI*((double)r.u*ll+(double)r.v*mm+(double)r.w*nn),&s,&c);
      vd+=(double)r.re*c-(double)r.im*s;
      sincos(2*M_PI*((double)r.u*lp+(double)r.v*mp),&s,&c);vp+=(double)r.re*c-(double)r.im*s;
      vr+=r.re;vi+=r.im;count++;
    }
    if(count){double s,c;sincos(2*M_PI*((double)gu[q]*lp+(double)gv[q]*mp),&s,&c);
      N[0]+=vd/count;N[1]+=vp/count;N[2]+=(vr*c-vi*s)/count;N[3]+=1;}
  }
  for(int k=0;k<4;k++)out[(size_t)k*np+p]=N[k];
}

static void rotate_pair(float& x,float& z,double angle){
  float a=x,b=z;x=(float)(std::cos(angle)*a+std::sin(angle)*b);
  z=(float)(-std::sin(angle)*a+std::cos(angle)*b);
}
static std::vector<ReplayRow> read_day(const std::string& root,int day,int& nside){
  std::vector<ReplayRow> all;int expected_t0=0;
  for(int seg=1;seg<=10;seg++){
    std::string path=root+"/day"+std::to_string(day)+"_seg"+std::to_string(seg)+".bin";
    std::ifstream in(path,std::ios::binary);ReplayHeader h{};in.read((char*)&h,sizeof(h));
    if(!in.good() || std::memcmp(h.magic,"PREPLAY1",8) || h.day!=day || h.seg!=seg || h.t0!=expected_t0 || h.nhalf!=28*h.tlen || h.forward_gpus!=1)
      throw std::runtime_error("cache identity/continuity mismatch: "+path);
    nside=h.nside;expected_t0+=h.tlen;size_t old=all.size();all.resize(old+h.nhalf);
    in.read((char*)(all.data()+old),(size_t)h.nhalf*sizeof(ReplayRow));
    if(!in.good())throw std::runtime_error("truncated cache "+path);
  }
  return all;
}

int main(int argc,char** argv){try{
  std::string gpu=get_arg(argc,argv,"--gpus","3");setenv("CUDA_VISIBLE_DEVICES",gpu.c_str(),1);
  CHECK_CUDA(cudaSetDevice(0));
  std::string cache=get_arg(argc,argv,"--cache_root","");
  std::string out=get_arg(argc,argv,"--out_dir","");
  std::string dcf_path=get_arg(argc,argv,"--dcf_bin","");
  int day=to_int(get_arg(argc,argv,"--day","1"),1),segs=to_int(get_arg(argc,argv,"--segs","10"),10);
  int np=to_int(get_arg(argc,argv,"--samples","32768"),32768);
  int only_segments=to_int(get_arg(argc,argv,"--segment_limit","0"),0);
  float scale=to_float(get_arg(argc,argv,"--grid_scale","1"),1);
  double angle=std::stod(get_arg(argc,argv,"--rotation","0"));
  if(cache.empty()||out.empty()||dcf_path.empty()||np<256||np%256||segs<1||!(scale>0))throw std::runtime_error("invalid replay arguments");
  ensure_dir(out);disk_guard(out);
  int nside=0;auto rows=read_day(cache,day,nside);int nt=(int)rows.size()/28;
  std::cout<<"REPLAY_PLAN day="<<day<<" half="<<rows.size()<<" samples="<<np<<" segs="<<segs<<" grid="<<scale<<" rotation="<<angle<<"\n";
  auto dcf=load_legacy_dcf(dcf_path);auto d_dcf=device_copy(dcf);
  auto original_drows=device_copy(rows);float* original_dweight=nullptr;
  CHECK_CUDA(cudaMalloc(&original_dweight,rows.size()*sizeof(float)));
  legacy_weights<<<((int)rows.size()+255)/256,256>>>(original_drows,(int)rows.size(),d_dcf,(int)dcf.size(),original_dweight);
  auto all_weights=host_copy(original_dweight,rows.size());
  CHECK_CUDA(cudaFree(original_dweight));CHECK_CUDA(cudaFree(original_drows));CHECK_CUDA(cudaFree(d_dcf));
  if(angle!=0)for(auto& row:rows){rotate_pair(row.u,row.w,angle);rotate_pair(row.x1,row.z1,angle);rotate_pair(row.x2,row.z2,angle);}

  auto ids=sample_ids(nside,np);auto did=device_copy(ids);
  float *dl=nullptr,*dm=nullptr,*dn=nullptr;
  CHECK_CUDA(cudaMalloc(&dl,np*sizeof(float)));CHECK_CUDA(cudaMalloc(&dm,np*sizeof(float)));CHECK_CUDA(cudaMalloc(&dn,np*sizeof(float)));
  // Use the original production sky-coordinate kernel, then gather directions.
  long long fullnp=12LL*nside*nside;float *fl=nullptr,*fm=nullptr,*fn=nullptr;
  CHECK_CUDA(cudaMalloc(&fl,fullnp*sizeof(float)));CHECK_CUDA(cudaMalloc(&fm,fullnp*sizeof(float)));CHECK_CUDA(cudaMalloc(&fn,fullnp*sizeof(float)));
  pix2lmn_nest_kernel<<<(fullnp+255)/256,256>>>(nside,0,(int)fullnp,fl,fm,fn);
  gather_samples<<<np/256,256>>>(did,np,0,fullnp,fl,dl);
  gather_samples<<<np/256,256>>>(did,np,0,fullnp,fm,dm);
  gather_samples<<<np/256,256>>>(did,np,0,fullnp,fn,dn);
  CHECK_CUDA(cudaFree(fl));CHECK_CUDA(cudaFree(fm));CHECK_CUDA(cudaFree(fn));CHECK_CUDA(cudaFree(did));
  auto hl=host_copy(dl,np),hm=host_copy(dm,np),hn=host_copy(dn,np);
  write_values(out+"/ids.u32",ids);write_values(out+"/l_original.f32",hl);write_values(out+"/m_original.f32",hm);write_values(out+"/n_original.f32",hn);
  if(angle!=0){for(int p=0;p<np;p++)rotate_pair(hl[p],hn[p],angle);
    CHECK_CUDA(cudaMemcpy(dl,hl.data(),np*sizeof(float),cudaMemcpyHostToDevice));CHECK_CUDA(cudaMemcpy(dn,hn.data(),np*sizeof(float),cudaMemcpyHostToDevice));}
  write_values(out+"/l.f32",hl);write_values(out+"/m.f32",hm);write_values(out+"/n.f32",hn);
  float R=1737.1e3f,h=300e3f;float theta=asinf(R/(R+h));float cosphi=cosf((float)M_PI-theta);
  std::vector<double> controls((size_t)np*9,0);
  std::vector<double> cells((size_t)np*4,0);
  std::array<std::vector<float>,3> prodN;std::array<std::vector<uint32_t>,3> prodW;
  for(int mode=0;mode<3;mode++){prodN[mode].assign(np,0);prodW[mode].assign(np,0);}
  std::ofstream meta(out+"/segments.csv");meta<<"seg,t0,tlen,half,groups,fa,fb,du,RES,fit_rms,fit_max,centroid_changed,replay_wall_s\n";
  int base=nt/segs,rem=nt%segs,t0=0;
  for(int s=0;s<segs;s++){
    int tlen=base+(s<rem),nh=tlen*28;std::vector<ReplayRow> part(rows.begin()+(size_t)t0*28,rows.begin()+(size_t)(t0+tlen)*28);
    std::vector<float> weights(all_weights.begin()+(size_t)t0*28,all_weights.begin()+(size_t)(t0+tlen)*28);
    HostTimer timer;timer.tic();
    std::array<std::vector<float>,11> full;
    for(auto& a:full)a.assign(nh*2,0);
    std::vector<float2> vis(nh);
    for(int k=0;k<nh;k++){
      int pos=half_to_full_pos_idx(k),neg=half_to_full_neg_idx(k);auto r=part[k];
      float xyz[11]={r.u,r.v,r.w,r.x1,r.y1,r.z1,r.inv1,r.x2,r.y2,r.z2,r.inv2};
      for(int j=0;j<11;j++)full[j][pos]=xyz[j];
      full[0][neg]=-r.u;full[1][neg]=-r.v;full[2][neg]=-r.w;
      for(int j=0;j<4;j++){full[3+j][neg]=xyz[7+j];full[7+j][neg]=xyz[3+j];}
      vis[k]=make_float2(r.re,r.im);
    }
    float fa=0,fb=0;double denom=0;compute_fa_fb_host_matlab_no_intercept(full[0].data(),full[1].data(),full[2].data(),nh*2,fa,fb,denom);
    double sum2=0,mx=0;for(auto r:part){double e=(double)r.w-(double)fa*r.u-(double)fb*r.v;sum2+=e*e;mx=std::max(mx,std::fabs(e));}
    float lmmax=std::max(std::sqrt(1.0f+fa*fa),std::sqrt(1.0f+fb*fb));
    float temp=std::ceil(2.0f*100e3f/30.0f*2.0f*lmmax);
    int baseRES=(int)(temp+2+1-std::fmod(temp,2.0f));int intervals=(int)std::ceil((double)(baseRES-1)/(double)scale);
    intervals=std::max(2,intervals);if(intervals&1)++intervals;int RES=intervals+1,half=intervals/2;
    float du=2.0f*100e3f/30.0f/(float)intervals;
    std::array<float*,11> dev;for(int j=0;j<11;j++)dev[j]=device_copy(full[j]);auto dv=device_copy(vis);
    GroupArtifactsHost grp;build_groups_device0_half(0,0,dev[0],dev[1],dv,nh,du,RES,half,1,0,grp);
    std::vector<float> gu(nh),gv(nh),ugu(grp.nuniq),vgu(grp.nuniq);
    std::vector<int> reps(grp.nuniq),centroid(grp.nuniq);int changed=0;
    for(int q=0;q<grp.nuniq;q++){
      int key=grp.keys_unique[q];if(key<=0)throw std::runtime_error("out-of-grid members: do not silently drop");
      int tmp=key-1,U=tmp/RES,V=tmp-U*RES;ugu[q]=(U-half)*du;vgu[q]=(V-half)*du;
      int lo=grp.offsets[q],hi=grp.offsets[q+1];reps[q]=grp.idx_sorted[lo+((hi-lo)>>1)];
      double uc=0,vc=0;for(int pos=lo;pos<hi;pos++){int k=grp.idx_sorted[pos];gu[k]=ugu[q];gv[k]=vgu[q];uc+=part[k].u;vc+=part[k].v;}
      uc/=(hi-lo);vc/=(hi-lo);double best=1e100;int winner=-1;
      for(int pos=lo;pos<hi;pos++){int k=grp.idx_sorted[pos];double d=(part[k].u-uc)*(part[k].u-uc)+(part[k].v-vc)*(part[k].v-vc);if(d<best||(d==best&&k<winner)){best=d;winner=k;}}
      centroid[q]=winner;changed+=(winner!=reps[q]);
    }
    auto drows=device_copy(part);auto dwgt=device_copy(weights);auto dgu=device_copy(gu);auto dgv=device_copy(gv);
    int part_size=8192,nparts=(nh+part_size-1)/part_size;double* partial=nullptr;
    CHECK_CUDA(cudaMalloc(&partial,(size_t)9*np*nparts*sizeof(double)));
    control_partial<<<dim3(np/256,nparts),256>>>(drows,dwgt,nh,dl,dm,dn,np,dgu,dgv,fa,fb,cosphi,part_size,nparts,partial);
    CHECK_CUDA(cudaPeekAtLastError());auto cpu=host_copy(partial,(size_t)9*np*nparts);
    for(int k=0;k<9;k++)for(int b=0;b<nparts;b++)for(int p=0;p<np;p++)controls[(size_t)k*np+p]+=cpu[((size_t)k*nparts+b)*np+p];
    if(s==0){std::vector<double> oracle;for(int k=0;k<9;k++)for(int p=0;p<16;p++)oracle.push_back(cpu[(size_t)k*nparts*np+p]);
      write_values(out+"/oracle_first_partition.f64",oracle);write_values(out+"/oracle_rows.bin",std::vector<ReplayRow>(part.begin(),part.begin()+std::min(part_size,nh)));
      write_values(out+"/oracle_weights.f32",std::vector<float>(weights.begin(),weights.begin()+std::min(part_size,nh)));
      write_values(out+"/oracle_gu.f32",std::vector<float>(gu.begin(),gu.begin()+std::min(part_size,nh)));
      write_values(out+"/oracle_gv.f32",std::vector<float>(gv.begin(),gv.begin()+std::min(part_size,nh)));
      std::ofstream cfg(out+"/oracle_config.txt");cfg<<std::setprecision(17)<<fa<<" "<<fb<<" "<<cosphi<<"\n";}
    CHECK_CUDA(cudaFree(partial));CHECK_CUDA(cudaFree(drows));CHECK_CUDA(cudaFree(dwgt));CHECK_CUDA(cudaFree(dgu));CHECK_CUDA(cudaFree(dgv));

    auto dk=device_copy(grp.keys_unique),doff=device_copy(grp.offsets),didx=device_copy(grp.idx_sorted);
    auto dugu=device_copy(ugu),dvgu=device_copy(vgu);float* dN=nullptr;uint32_t* dW=nullptr;
    auto cell_rows=device_copy(part);double* dcell=nullptr;CHECK_CUDA(cudaMalloc(&dcell,(size_t)4*np*sizeof(double)));
    cell_controls<<<np/256,256>>>(cell_rows,doff,didx,dugu,dvgu,grp.nuniq,dl,dm,dn,np,fa,fb,cosphi,dcell);
    CHECK_CUDA(cudaPeekAtLastError());auto hcell=host_copy(dcell,(size_t)4*np);
    for(size_t k=0;k<cells.size();k++)cells[k]+=hcell[k];CHECK_CUDA(cudaFree(dcell));CHECK_CUDA(cudaFree(cell_rows));
    CHECK_CUDA(cudaMalloc(&dN,np*sizeof(float)));CHECK_CUDA(cudaMalloc(&dW,np*sizeof(uint32_t)));
    float2* repV=nullptr;float4 *p1=nullptr,*p2=nullptr;
    CHECK_CUDA(cudaMalloc(&repV,grp.nuniq*sizeof(float2)));CHECK_CUDA(cudaMalloc(&p1,grp.nuniq*sizeof(float4)));CHECK_CUDA(cudaMalloc(&p2,grp.nuniq*sizeof(float4)));
    for(int mode=0;mode<3;mode++){
      if(mode==0){
        recon_seg_blockage_half<64><<<np/256,256>>>(np,dl,dm,dn,dk,doff,didx,dv,
          dev[3],dev[4],dev[5],dev[6],dev[7],dev[8],dev[9],dev[10],grp.nuniq,RES,half,du,fa,fb,cosphi,0,dN,dW);
      }else{
        auto dr=device_copy(mode==1?reps:centroid);
        gather_rep_half_inputs<<<(grp.nuniq+255)/256,256>>>(dr,grp.nuniq,dv,dev[3],dev[4],dev[5],dev[6],dev[7],dev[8],dev[9],dev[10],repV,p1,p2);
        recon_seg_blockage_half_rep<64><<<np/256,256>>>(np,dl,dm,dn,dk,dugu,dvgu,repV,p1,p2,grp.nuniq,fa,fb,cosphi,dN,dW);
        CHECK_CUDA(cudaFree(dr));
      }
      CHECK_CUDA(cudaPeekAtLastError());auto nN=host_copy(dN,np);auto nW=host_copy(dW,np);
      for(int p=0;p<np;p++){prodN[mode][p]+=nN[p];prodW[mode][p]+=nW[p];}
    }
    CHECK_CUDA(cudaFree(dN));CHECK_CUDA(cudaFree(dW));CHECK_CUDA(cudaFree(repV));CHECK_CUDA(cudaFree(p1));CHECK_CUDA(cudaFree(p2));
    CHECK_CUDA(cudaFree(dk));CHECK_CUDA(cudaFree(doff));CHECK_CUDA(cudaFree(didx));CHECK_CUDA(cudaFree(dugu));CHECK_CUDA(cudaFree(dvgu));CHECK_CUDA(cudaFree(dv));
    for(auto ptr:dev)CHECK_CUDA(cudaFree(ptr));
    meta<<std::setprecision(12)<<s+1<<","<<t0<<","<<tlen<<","<<nh<<","<<grp.nuniq<<","<<fa<<","<<fb<<","<<du<<","<<RES<<","<<std::sqrt(sum2/nh)<<","<<mx<<","<<changed<<","<<timer.toc_s()<<"\n";meta.flush();
    std::cout<<"REPLAY_SEG_DONE "<<s+1<<"/"<<segs<<" wall="<<timer.toc_s()<<"s groups="<<grp.nuniq<<"\n";std::cout.flush();
    t0+=tlen;if(only_segments && s+1>=only_segments)break;
  }
  const char* labels[9]={"direct_N","plane_N","quant_member_N","raw_nm1_N","legacy_dcf_direct_N","legacy_dcf_plane_N","legacy_dcf_quant_N","natural_W","legacy_dcf_W"};
  for(int k=0;k<9;k++)write_values(out+"/"+labels[k]+".f64",std::vector<double>(controls.begin()+(size_t)k*np,controls.begin()+(size_t)(k+1)*np));
  const char* clabels[4]={"cell_direct_N","cell_plane_N","cell_quant_N","cell_W"};
  for(int k=0;k<4;k++)write_values(out+"/"+clabels[k]+".f64",std::vector<double>(cells.begin()+(size_t)k*np,cells.begin()+(size_t)(k+1)*np));
  const char* prod[3]={"groupmean","rep","centroid"};for(int mode=0;mode<3;mode++){
    write_values(out+"/"+prod[mode]+"_N.f32",prodN[mode]);write_values(out+"/"+prod[mode]+"_W.u32",prodW[mode]);}
  std::cout<<"REPLAY_DONE samples="<<np<<" day="<<day<<" segs="<<segs<<" segment_limit="<<only_segments<<"\n";
  return 0;
}catch(const std::exception& error){std::cerr<<"ERROR "<<error.what()<<"\n";return 2;}}
