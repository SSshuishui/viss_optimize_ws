#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
namespace prodck {
namespace fs=std::filesystem;
constexpr uint64_t MAGIC=0x50524f44434b3031ULL;
#pragma pack(push,1)
struct Header {
 uint64_t magic=MAGIC,version=1;
 char identity[65]{};
 uint64_t geometry_hash=0,npix=0,nside=0,day=0,seed=0,t0=0,tlen=0;
 uint64_t mode=0,gi=0,G=0,pix0=0,n_chunk=0,slab=0,next=0,nslabs=0,N_half=0;
};
#pragma pack(pop)
static_assert(sizeof(Header)==209,"fixed packed checkpoint header");
inline uint64_t fnv(const void* data,size_t size,uint64_t h=1469598103934665603ULL){auto p=(const unsigned char*)data;for(size_t k=0;k<size;k++){h^=p[k];h*=1099511628211ULL;}return h;}
inline size_t payload_bytes(const Header&h){if(h.mode!=0&&h.mode!=4)throw std::runtime_error("checkpoint mode");return h.N_half*(h.mode==4?16:8);}
inline void progress(const Header&h){
 if(h.magic!=MAGIC||h.version!=1||!h.slab||!h.n_chunk||!h.N_half||h.next>h.n_chunk||h.slab%256||h.next%256||h.nslabs>(h.n_chunk+h.slab-1)/h.slab||h.next!=std::min(h.nslabs*h.slab,h.n_chunk)||h.nslabs!=(h.next+h.slab-1)/h.slab)throw std::runtime_error("checkpoint progress");
 if(h.mode==4&&(h.slab%32768||h.n_chunk%32768||h.next%32768))throw std::runtime_error("checkpoint microchunk alignment");
}
inline void write(const fs::path&path,const Header&h,const void*payload,size_t bytes){
 progress(h);if(bytes!=payload_bytes(h))throw std::runtime_error("checkpoint payload size");fs::path tmp=path.string()+".tmp";uint64_t hash=fnv(payload,bytes,fnv(&h,sizeof(h)));
 {std::ofstream f(tmp,std::ios::binary);f.write((const char*)&h,sizeof(h));f.write((const char*)payload,bytes);f.write((const char*)&hash,sizeof(hash));f.flush();if(!f.good())throw std::runtime_error("checkpoint write");}
 int fd=::open(tmp.c_str(),O_RDONLY);if(fd<0)throw std::runtime_error("checkpoint open for sync");if(::fsync(fd)){::close(fd);throw std::runtime_error("checkpoint fsync");}::close(fd);fs::rename(tmp,path);
 int dir=::open(path.parent_path().c_str(),O_RDONLY|O_DIRECTORY);if(dir>=0){::fsync(dir);::close(dir);}
}
struct Read {Header header;std::vector<unsigned char> payload;};
inline Read read(const fs::path&path,const Header&wanted){
 Read r;std::ifstream f(path,std::ios::binary);f.read((char*)&r.header,sizeof(Header));if(!f.good())throw std::runtime_error("short checkpoint header");auto h=r.header;Header expected=wanted;expected.next=h.next;expected.nslabs=h.nslabs;
 if(std::memcmp(&h,&expected,sizeof(Header)))throw std::runtime_error("checkpoint identity drift");progress(h);size_t bytes=payload_bytes(h);r.payload.resize(bytes);f.read((char*)r.payload.data(),bytes);uint64_t stored=0;f.read((char*)&stored,sizeof(stored));if(!f.good()||f.peek()!=std::char_traits<char>::eof()||stored!=fnv(r.payload.data(),bytes,fnv(&h,sizeof(h))))throw std::runtime_error("checkpoint checksum/EOF");return r;
}
}
