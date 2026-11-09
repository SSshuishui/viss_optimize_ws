#pragma once
#include "checkpoint_io.hpp"
#include <openssl/sha.h>
#include <array>
#include <csignal>
#include <filesystem>
#include <mutex>
#include <sys/resource.h>
namespace prod {
namespace fs=std::filesystem;
inline volatile std::sig_atomic_t stopped=0;
inline std::string pause_path;
inline std::mutex files_mutex;
inline void stop_handler(int){stopped=1;}
inline bool pause(){return stopped||(!pause_path.empty()&&fs::exists(pause_path));}
inline std::string hex(const unsigned char*d){std::ostringstream o;for(int i=0;i<32;i++)o<<std::hex<<std::setw(2)<<std::setfill('0')<<(int)d[i];return o.str();}
inline std::string sha(const void*p,size_t n){unsigned char d[32];SHA256((const unsigned char*)p,n,d);return hex(d);}
inline std::string file_sha(const fs::path&p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("SHA input missing");SHA256_CTX h;SHA256_Init(&h);std::vector<char>b(1<<20);while(f){f.read(b.data(),b.size());if(f.gcount())SHA256_Update(&h,b.data(),f.gcount());}if(!f.eof())throw std::runtime_error("SHA read");unsigned char d[32];SHA256_Final(d,&h);return hex(d);}
inline void blob(const fs::path&p,const void*v,size_t n){auto tmp=p.string()+".tmp";{std::ofstream f(tmp,std::ios::binary);f.write((const char*)v,n);f.flush();if(!f)throw std::runtime_error("artifact write");}int fd=::open(tmp.c_str(),O_RDONLY);if(fd<0||::fsync(fd))throw std::runtime_error("artifact fsync");::close(fd);fs::rename(tmp,p);}
inline void text(const fs::path&p,const std::string&s){blob(p,s.data(),s.size());}
template<class T>inline std::vector<T>read(const fs::path&p,size_t n){std::ifstream f(p,std::ios::binary);std::vector<T>v(n);f.read((char*)v.data(),n*sizeof(T));if(!f.good()||f.peek()!=EOF)throw std::runtime_error("reference size/EOF");return v;}
inline void require(bool pass,const char*why){if(!pass)throw std::runtime_error(why);}
struct Record{uint32_t half_id,full_id,time,pair;float u,v,w,x1,y1,z1,in1,x2,y2,z2,in2,cosphi;};
struct Geometry{uint64_t magic;uint32_t nside,day,t0,tlen,nrec;float fa,fb,du,cosphi;int32_t RES,half;};
static_assert(sizeof(Geometry)==56&&sizeof(Record)==64,"reference format");
struct References{
 Geometry g{};std::vector<Record>r;std::vector<int>keys,ids;std::vector<double2>all,pilot;
 explicit References(const fs::path&dir){std::ifstream f(dir/"rep_geometry.bin",std::ios::binary);f.read((char*)&g,sizeof(g));require(f.good()&&g.magic==0x534c414247454f32ULL&&g.nside==16384&&g.day==1&&g.t0==0&&g.tlen==20944&&g.nrec==86668,"reference geometry header");r.resize(g.nrec);f.read((char*)r.data(),r.size()*sizeof(Record));require(f.good()&&f.peek()==EOF,"reference geometry EOF");keys=read<int>(dir/"full_keys.i32",g.nrec);ids=read<int>(dir/"rep_half_idx.i32",g.nrec);all=read<double2>(dir/"MODE4_G3_p0_allrep_raw.f64x2",g.nrec);auto progress=read<double2>(dir/"MODE4_G3_p0_small_audit8_progress.f64x2",32*8);pilot.assign(progress.begin(),progress.begin()+8);for(size_t q=0;q<r.size();q++)require((int)r[q].half_id==ids[q]&&r[q].full_id==r[q].half_id/28*56+r[q].half_id%28,"frozen representative map");}
 uint64_t geometry(const OrbitSegSlot&s,float cosphi)const{
  require(s.t0==0&&s.tlen==20944&&s.segN==1172864,"first original segment identity");
  for(const auto&a:r){int k=a.full_id;const float got[]={s.h_u[k],s.h_v[k],s.h_w[k],s.h_x1[k],s.h_y1[k],s.h_z1[k],s.h_x2[k],s.h_y2[k],s.h_z2[k],cosphi};const float expected[]={a.u,a.v,a.w,a.x1,a.y1,a.z1,a.x2,a.y2,a.z2,a.cosphi};require(!std::memcmp(got,expected,sizeof(got)),"original representative geometry changed");}
  uint64_t h=1469598103934665603ULL;for(const float*p:{s.h_u,s.h_v,s.h_w,s.h_x1,s.h_y1,s.h_z1,s.h_x2,s.h_y2,s.h_z2})h=prodck::fnv(p,(size_t)s.segN*4,h);return h;
 }
 void groups(const GroupArtifactsHost&group,float fa,float fb,float du,int RES,int half)const{
  require(group.nuniq==86668&&group.keys_unique==keys&&group.rep_half_idx==ids,"full original CUB group/representative identity");
  require(!std::memcmp(&fa,&g.fa,4)&&!std::memcmp(&fb,&g.fb,4)&&!std::memcmp(&du,&g.du,4)&&RES==g.RES&&half==g.half,"original STS fit/grid changed");
 }
 void state(const std::vector<double2>&v,const fs::path&out,bool first_slab)const{
  std::vector<double2>actual;bool equal=true;if(first_slab){for(int j=0;j<8;j++){size_t q=((2ULL*j+1)*r.size())/16;actual.push_back(v[ids[q]]);}equal=!std::memcmp(actual.data(),pilot.data(),8*sizeof(double2));}
  else{actual.resize(ids.size());for(size_t q=0;q<ids.size();q++)actual[q]=v[ids[q]];equal=!std::memcmp(actual.data(),all.data(),all.size()*sizeof(double2));}
  blob(out.string()+".f64x2",actual.data(),actual.size()*sizeof(double2));text(out.string()+".json",std::string("{\"pass\":")+(equal?"true":"false")+",\"bitwise\":"+(equal?"true":"false")+",\"baselines\":"+std::to_string(actual.size())+",\"input_pixels\":"+(first_slab?"33554432":"1073741824")+",\"scope\":\"same global input prefix before partition cast; not cross-G final V\"}\n");require(equal,"validated MODE4 same-prefix double state differs");
 }
};
inline void memory(const fs::path&dir,int G,bool require_reserve){
 std::ostringstream cards;bool ok=true;for(int gi=0;gi<G;gi++){CHECK_CUDA(cudaSetDevice(gi));CHECK_CUDA(cudaDeviceSynchronize());prodalloc::observe(gi,0,cudaSuccess);auto m=prodalloc::minfree[gi].load();ok=ok&&m>=(2ULL<<30);if(gi)cards<<',';cards<<"{\"logical\":"<<gi<<",\"peak_used_bytes\":"<<prodalloc::peak[gi].load()<<",\"minimum_free_bytes\":"<<m<<'}';}std::ostringstream o;o<<"{\"pass\":"<<(ok?"true":"false")<<",\"minimum_free_required_bytes\":2147483648,\"cards\":["<<cards.str()<<"],\"reserve_pass\":"<<(ok?"true":"false")<<"}\n";text(dir/"memory.json",o.str());if(require_reserve)require(ok,"actual allocation free reserve below 2GiB");
}
#pragma pack(push,1)
struct ForwardCommit{uint64_t magic=0x465744434f4d3031ULL;char identity[65]{},raw_sha[65]{};uint64_t mode=0,G=0,slab=0,geometry=0,npix=0,t0=0,tlen=0,N_half=0;};
#pragma pack(pop)
inline ForwardCommit forward_identity(const std::string&id,int mode,int G,uint64_t slab,uint64_t geom){ForwardCommit c{};std::memcpy(c.identity,id.c_str(),65);c.mode=mode;c.G=G;c.slab=slab;c.geometry=geom;c.npix=3221225472ULL;c.t0=0;c.tlen=20944;c.N_half=586432;return c;}
inline void commit_forward(const fs::path&dir,ForwardCommit c,const std::vector<float2>&v){require(v.size()==c.N_half,"full forward commit size");blob(dir/"forward_raw.f32x2",v.data(),v.size()*8);auto digest=file_sha(dir/"forward_raw.f32x2");std::memcpy(c.raw_sha,digest.c_str(),65);std::ostringstream o;o<<"{\"execution_identity\":\""<<c.identity<<"\",\"mode\":"<<c.mode<<",\"G\":"<<c.G<<",\"slab\":"<<c.slab<<",\"geometry_hash\":"<<c.geometry<<",\"npix\":"<<c.npix<<",\"N_half\":"<<c.N_half<<",\"t0\":0,\"tlen\":20944,\"raw_sha256\":\""<<digest<<"\"}\n";text(dir/"forward_committed.json",o.str());blob(dir/"forward_committed.bin",&c,sizeof(c));}
inline std::vector<float2>load_forward(const fs::path&dir,ForwardCommit expected){auto v=read<ForwardCommit>(dir/"forward_committed.bin",1);std::memcpy(expected.raw_sha,v[0].raw_sha,65);require(!std::memcmp(&expected,&v[0],sizeof(expected)),"committed forward execution/mode/G/slab/geometry identity drift");require(file_sha(dir/"forward_raw.f32x2")==std::string(v[0].raw_sha,64),"committed full forward SHA mismatch");return read<float2>(dir/"forward_raw.f32x2",expected.N_half);}
#pragma pack(push,1)
struct Cleanup{uint64_t gi=0,bytes=0;char sha256[65]{};};
#pragma pack(pop)
inline void checkpoint_cleanup(const fs::path&dir,int G){
 auto registry=dir/"checkpoint_cleanup.bin";std::vector<Cleanup> rows;
 if(fs::exists(registry))rows=read<Cleanup>(registry,G);
 else{rows.resize(G);std::ostringstream o;o<<"{\"registered_before_delete\":[";for(int gi=0;gi<G;gi++){if(gi)o<<',';auto p=dir/("forward_gpu"+std::to_string(gi)+".ck");require(fs::exists(p),"completed forward checkpoint missing before cleanup registration");rows[gi].gi=gi;rows[gi].bytes=fs::file_size(p);auto digest=file_sha(p);std::memcpy(rows[gi].sha256,digest.c_str(),65);o<<"{\"path\":\""<<p.filename().string()<<"\",\"bytes\":"<<rows[gi].bytes<<",\"sha256\":\""<<digest<<"\"}";}o<<"],\"reason\":\"full forward V atomically committed; inverse replay uses committed V\"}\n";text(dir/"checkpoint_cleanup.json",o.str());blob(registry,rows.data(),rows.size()*sizeof(Cleanup));}
 for(int gi=0;gi<G;gi++){require(rows[gi].gi==(uint64_t)gi,"cleanup GPU identity");auto p=dir/("forward_gpu"+std::to_string(gi)+".ck");if(fs::exists(p)){require(fs::file_size(p)==rows[gi].bytes&&file_sha(p)==std::string(rows[gi].sha256,64),"registered checkpoint changed before deletion");fs::remove(p);}}
}
inline void samples(std::vector<GpuCtx>&ctx,long long npix,long long chunk,const fs::path&dir,const std::string&field){std::vector<uint32_t>ids(2048),v(2048);for(int k=0;k<2048;k++){uint64_t p=((2ULL*k+1)*npix)/4096;ids[k]=p;int gi=p/chunk;uint64_t local=p-ctx[gi].pix0;CHECK_CUDA(cudaSetDevice(gi));const void*src=field=="W"?(void*)(ctx[gi].d_Wacc+local):(void*)(ctx[gi].d_Cacc+local);CHECK_CUDA(cudaMemcpy(&v[k],src,4,cudaMemcpyDeviceToHost));}blob(dir/"sample_ids.u32",ids.data(),ids.size()*4);blob(dir/("sample_"+field+(field=="W"?".u32":".f32")),v.data(),v.size()*4);}
inline void scan(std::vector<GpuCtx>&ctx,const fs::path&dir,const std::string&field){SHA256_CTX h;SHA256_Init(&h);uint64_t count=0,nonfinite=0,zero=0;long double sum=0;double lo=INFINITY,hi=-INFINITY;std::vector<uint32_t>v(1<<20);for(size_t gi=0;gi<ctx.size();gi++){CHECK_CUDA(cudaSetDevice(gi));for(long long off=0;off<ctx[gi].n_chunk;off+=v.size()){size_t n=std::min<long long>(v.size(),ctx[gi].n_chunk-off);const void*src=field=="W"?(void*)(ctx[gi].d_Wacc+off):(void*)(ctx[gi].d_Cacc+off);CHECK_CUDA(cudaMemcpy(v.data(),src,n*4,cudaMemcpyDeviceToHost));SHA256_Update(&h,v.data(),n*4);for(size_t k=0;k<n;k++){float f;std::memcpy(&f,&v[k],4);double x=field=="W"?(double)v[k]:(double)f;nonfinite+=!std::isfinite(x);zero+=x==0;if(std::isfinite(x)){lo=std::min(lo,x);hi=std::max(hi,x);sum+=x;}}count+=n;if(pause())throw std::runtime_error("pause during full field scan; replay inverse on explicit resume");}}unsigned char d[32];SHA256_Final(d,&h);std::ostringstream o;o<<std::setprecision(20)<<"{\"pixels\":"<<count<<",\"bytes\":"<<count*4<<",\"sha256\":\""<<hex(d)<<"\",\"nonfinite\":"<<nonfinite<<",\"zero\":"<<zero<<",\"min\":"<<lo<<",\"max\":"<<hi<<",\"sum\":"<<(double)sum<<",\"order\":\"global NEST logical GPU0 then GPU1; all pixels\"}\n";text(dir/("full_"+field+".json"),o.str());require(count==3221225472ULL&&!nonfinite,"full-domain pixel/finite gate");}
inline void final_info(const fs::path&dir,int mode,int G,uint64_t slab,double elapsed){struct rusage r{};getrusage(RUSAGE_SELF,&r);std::ostringstream o;o<<std::setprecision(12)<<"{\"state\":\"completed\",\"mode\":"<<mode<<",\"G\":"<<G<<",\"slab\":"<<slab<<",\"npix\":3221225472,\"input_pixels\":3221225472,\"output_pixels\":3221225472,\"all_half_baselines\":586432,\"original_segments\":10,\"executed_segments\":1,\"groups\":86668,\"max_rss_KiB\":"<<r.ru_maxrss<<",\"process_internal_wall_s\":"<<elapsed<<",\"inverse\":\"original unmodified midpoint representative STS N/W\",\"maps_written\":false}\n";text(dir/"production.json",o.str());}
__global__ void cast_state(const double2*s,float2*v,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)v[i]=make_float2((float)s[i].x,(float)s[i].y);}
}
