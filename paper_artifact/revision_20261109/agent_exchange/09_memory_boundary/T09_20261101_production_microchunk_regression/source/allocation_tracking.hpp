#pragma once
#include <atomic>
#include <mutex>
#include <limits>
namespace prodalloc {
inline std::atomic<size_t> peak[2]={{0},{0}};
inline std::atomic<size_t> minfree[2]={{std::numeric_limits<size_t>::max()},{std::numeric_limits<size_t>::max()}};
inline std::mutex log_mutex;
inline std::ofstream log;
inline void observe(int dev,size_t bytes,cudaError_t rc){
 size_t free=0,total=0;cudaMemGetInfo(&free,&total);if(dev>=0&&dev<2){size_t value=peak[dev].load();while(value<total-free&&!peak[dev].compare_exchange_weak(value,total-free)){}value=minfree[dev].load();while(value>free&&!minfree[dev].compare_exchange_weak(value,free)){}}
 std::lock_guard<std::mutex>guard(log_mutex);if(log.good())log<<dev<<','<<bytes<<','<<(int)rc<<','<<free<<','<<total<<'\n'<<std::flush;
}
template<class T>inline cudaError_t allocate(T**out,size_t bytes){int dev=-1;cudaGetDevice(&dev);cudaError_t rc=cudaMalloc((void**)out,bytes);observe(dev,bytes,rc);return rc;}
}
// Includes of CUB were completed by the unchanged production common header.
// Only subsequent explicit host allocations are wrapped; no kernel body changes.
#define cudaMalloc prodalloc::allocate
