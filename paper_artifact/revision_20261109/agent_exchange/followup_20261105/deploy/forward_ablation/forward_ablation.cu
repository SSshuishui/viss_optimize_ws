#include "orbit_online_mirror_tile_vissfast.cuh"
#include "viss_recon_kernel_mirror_tile_vissfast.cuh"
#include "point_only.cuh"

#include <array>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace fs = std::filesystem;
constexpr int kNside = 4096;
constexpr int kPixels = 12 * kNside * kNside;
constexpr int kStart = 0;
constexpr int kTimes = 1280;
constexpr int kDay = 1;
constexpr uint64_t kSeed = 42;
constexpr float kLambda = 30.0f;
constexpr int kSigned = kTimes * SIGNED_BASELINES_PER_T;
constexpr int kHalf = kTimes * UNIQUE_BASELINES_PER_T;
constexpr int kTile = 256;
constexpr int kTiles = (kPixels + kTile - 1) / kTile;
constexpr int kBlock = 256;
constexpr int kBlocks = (kHalf + kBlock - 1) / kBlock;
constexpr size_t kShared = 5 * kTile * sizeof(float);
constexpr int kPairs = 4;
constexpr int kGeometryPairs = 8;
constexpr size_t kGeometryBytes = 9 * size_t(kSigned) * sizeof(float);
constexpr double kTileTolerance = 1e-5;

class DeviceBuffers {
  std::vector<void*> ptrs_;
public:
  template<class T> T* get(size_t count) {
    T* ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&ptr, count * sizeof(T)));
    ptrs_.push_back(ptr);
    return ptr;
  }
  ~DeviceBuffers() { for (void* ptr : ptrs_) cudaFree(ptr); }
};

template<class T> void write_array(const fs::path& path, const std::vector<T>& v) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
  if (!out) throw std::runtime_error("Cannot write " + path.string());
}

std::ofstream csv(const fs::path& path, const char* header) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("Cannot write " + path.string());
  out << std::setprecision(17) << header << '\n';
  return out;
}

struct PairError {
  double relative_l2, max_over_peak, max_abs;
  size_t changed;
};

PairError compare(const std::vector<float2>& a, const std::vector<float2>& b) {
  double diff2 = 0, ref2 = 0, peak = 0, max_abs = 0;
  size_t changed = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i].x) || !std::isfinite(a[i].y) ||
        !std::isfinite(b[i].x) || !std::isfinite(b[i].y))
      throw std::runtime_error("Non-finite raw visibility");
    const double d = std::hypot(double(a[i].x) - b[i].x, double(a[i].y) - b[i].y);
    const double r = std::hypot(double(b[i].x), double(b[i].y));
    diff2 += d*d;
    ref2 += r*r;
    peak = std::max(peak, r);
    max_abs = std::max(max_abs, d);
    changed += std::memcmp(&a[i], &b[i], sizeof(float2)) != 0;
  }
  if (ref2 <= 0 || peak <= 0) throw std::runtime_error("Zero point-only reference");
  return {std::sqrt(diff2 / ref2), max_abs / peak, max_abs, changed};
}

int main(int argc, char** argv) try {
  if (argc != 3) throw std::runtime_error("Usage: forward_ablation SKY_FP32_BIN NEW_RUN_DIRECTORY");
  const fs::path sky_path(argv[1]), out(argv[2]);
  if (!fs::is_directory(out)) throw std::runtime_error("Run directory missing");
  if (fs::file_size(sky_path) != size_t(kPixels) * sizeof(float))
    throw std::runtime_error("Wrong sky binary size");
  if (fs::exists(out / "timings.csv")) throw std::runtime_error("Refusing to overwrite run evidence");
  int devices = 0;
  CHECK_CUDA(cudaGetDeviceCount(&devices));
  if (devices != 1) throw std::runtime_error("Require exactly one visible GPU");
  CHECK_CUDA(cudaSetDevice(0));
  cudaDeviceProp device{};
  CHECK_CUDA(cudaGetDeviceProperties(&device, 0));
  if (device.major != 8 || device.minor != 9)
    throw std::runtime_error("This build and protocol expect an sm_89 GPU");

  std::cout << "device=" << device.name << " pixels=" << kPixels << " day=" << kDay
            << " timestamp_start=" << kStart << " timestamps=" << kTimes
            << " half_baselines=" << kHalf << " blocks=" << kBlocks << std::endl;
  DeviceBuffers dev;
  cudaStream_t stream;
  cudaEvent_t start, finish;
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&finish));

  // Exact production inputs and launch geometry. Sky pixels stay resident.
  float* B = dev.get<float>(kPixels);
  float* l = dev.get<float>(kPixels);
  float* m = dev.get<float>(kPixels);
  float* n = dev.get<float>(kPixels);
  float* nm1 = dev.get<float>(kPixels);
  std::array<float*,5> cone;
  for (auto& ptr : cone) ptr = dev.get<float>(kTiles);
  {
    std::vector<float> sky(kPixels);
    if (!load_single_bin_exact(sky_path.string(), sky.data(), kPixels))
      throw std::runtime_error("Cannot read complete production sky");
    const double area = 4.0 * M_PI / double(kPixels);
    #pragma omp parallel for
    for (int i = 0; i < kPixels; ++i) sky[i] = float(double(sky[i]) * area);
    CHECK_CUDA(cudaMemcpyAsync(B, sky.data(), sky.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }
  pix2lmn_nest_kernel<<<kTiles,kTile,0,stream>>>(kNside, 0, kPixels, l, m, n);
  build_nm1_kernel<<<kTiles,kTile,0,stream>>>(n, nm1, kPixels);
  build_tile_cone_meta_kernel<kTile><<<kTiles,kBlock,0,stream>>>(
      l, m, n, kPixels, cone[0], cone[1], cone[2], cone[3], cone[4], kTiles);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaStreamSynchronize(stream));

  HostTimer host;
  host.tic();
  OrbitGenCtx orbit{};
  orbit_gen_init(orbit, 0, kLambda, kDay, kTimes, kSeed);
  const double orbit_init_s = host.toc_s();
  if (kStart + kTimes > orbit.segLen)
    throw std::runtime_error("Window must stay inside first retained orbit window");

  // Order: u,v,w,x1,y1,z1,x2,y2,z2, invnorm1,invnorm2.
  std::array<float*,11> geometry;
  for (auto& ptr : geometry) ptr = dev.get<float>(kSigned);
  float* positions = dev.get<float>(size_t(kTimes) * 3 * SATNUM);
  const int signed_blocks = (kSigned + kBlock - 1) / kBlock;
  auto norms = [&]() {
    invnorm3_kernel<<<signed_blocks,kBlock,0,stream>>>(
        geometry[3], geometry[4], geometry[5], geometry[9], kSigned);
    invnorm3_kernel<<<signed_blocks,kBlock,0,stream>>>(
        geometry[6], geometry[7], geometry[8], geometry[10], kSigned);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));
  };
  auto online = [&]() {
    k_gather_pos_range<<<(kTimes+kBlock-1)/kBlock,kBlock,0,stream>>>(
        orbit.d_x, orbit.d_y, orbit.d_z, positions, orbit.d_t1s,
        orbit.OrbitRes, orbit.ProcessionCount, orbit.segLen, kStart, kTimes);
    k_compute_uvw_xyz<<<signed_blocks,kBlock,0,stream>>>(
        positions, kTimes, kLambda, geometry[0], geometry[1], geometry[2],
        geometry[3], geometry[4], geometry[5], geometry[6], geometry[7], geometry[8]);
    CHECK_CUDA(cudaGetLastError());
    norms();
  };
  online();
  std::vector<float> snapshot(geometry.size() * size_t(kSigned));
  auto read_geometry = [&](std::vector<float>& into) {
    for (size_t j = 0; j < geometry.size(); ++j)
      CHECK_CUDA(cudaMemcpy(into.data() + j*kSigned, geometry[j],
                            kSigned*sizeof(float), cudaMemcpyDeviceToHost));
  };
  read_geometry(snapshot);
  if (!validate_uvw_halfsym_layout_ptr(snapshot.data(), snapshot.data()+kSigned,
                                      snapshot.data()+2*kSigned, kSigned))
    throw std::runtime_error("Generated signed/half layout failed");
  const fs::path snapshot_path = out / "geometry_snapshot.f32";
  {
    std::ofstream raw(snapshot_path, std::ios::binary);
    raw.write(reinterpret_cast<const char*>(snapshot.data()), kGeometryBytes);
    if (!raw) throw std::runtime_error("Cannot write prerecorded geometry snapshot");
  }

  // The file is generated once from exactly the same model. Each replay reads
  // its nine raw arrays through the ordinary (warm) filesystem page cache.
  float* staging = nullptr;
  CHECK_CUDA(cudaMallocHost(&staging, kGeometryBytes));
  auto replay = [&]() {
    std::ifstream raw(snapshot_path, std::ios::binary);
    raw.read(reinterpret_cast<char*>(staging), kGeometryBytes);
    if (!raw) throw std::runtime_error("Cannot read prerecorded geometry snapshot");
    for (size_t j = 0; j < 9; ++j)
      CHECK_CUDA(cudaMemcpyAsync(geometry[j], staging+j*kSigned, kSigned*sizeof(float),
                                cudaMemcpyHostToDevice, stream));
    norms();
  };
  auto geom_csv = csv(out / "geometry_timings.csv",
                     "pair,order,position,mode,wall_s,byte_equal_to_snapshot");
  std::vector<float> check(snapshot.size());
  // Initial generation and initial replay are warmups, excluded from timings.
  replay();
  read_geometry(check);
  if (check != snapshot || std::memcmp(check.data(), snapshot.data(),
                                      snapshot.size()*sizeof(float)) != 0)
    throw std::runtime_error("Warm replay geometry differs");
  for (int pair = 1; pair <= kGeometryPairs; ++pair) {
    const bool online_first = pair % 2 == 1;
    for (int i = 0; i < 2; ++i) {
      const bool use_online = (i == 0) == online_first;
      host.tic();
      if (use_online) online(); else replay();
      const double wall = host.toc_s();
      read_geometry(check);
      const bool equal = std::memcmp(check.data(), snapshot.data(),
                                     snapshot.size()*sizeof(float)) == 0;
      geom_csv << pair << ',' << (online_first ? "OR" : "RO") << ',' << i+1 << ','
               << (use_online ? "online_ready" : "cached_file_to_device") << ','
               << wall << ',' << equal << std::endl;
      if (!equal) throw std::runtime_error("Geometry differs from frozen comparison bytes");
    }
  }

  const float R = 1737.1e3f, h = 300e3f;
  const float cosphi = cosf(float(M_PI) - asinf(R / (R + h)));
  float2* V = dev.get<float2>(kHalf);
  const auto launch = [&](bool fast) {
    CHECK_CUDA(cudaEventRecord(start, stream));
    // Shared allocation, indexing, storage and arithmetic are identical.
    if (fast) {
      viss_partial_all_halfsym_tilecone<kTile,true><<<kBlocks,kBlock,kShared,stream>>>(
          B,l,m,n,nm1,kPixels,cone[0],cone[1],cone[2],cone[3],cone[4],kTiles,
          geometry[0],geometry[1],geometry[2],geometry[3],geometry[4],geometry[5],geometry[9],
          geometry[6],geometry[7],geometry[8],geometry[10],kHalf,cosphi,V);
    } else {
      viss_partial_all_halfsym_point_only<kTile,true><<<kBlocks,kBlock,kShared,stream>>>(
          B,l,m,n,nm1,kPixels,cone[0],cone[1],cone[2],cone[3],cone[4],kTiles,
          geometry[0],geometry[1],geometry[2],geometry[3],geometry[4],geometry[5],geometry[9],
          geometry[6],geometry[7],geometry[8],geometry[10],kHalf,cosphi,V);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(finish, stream));
    CHECK_CUDA(cudaEventSynchronize(finish));
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, finish));
    return double(ms) / 1000.0;
  };
  auto timing = csv(out / "timings.csv", "pair,order,position,variant,kernel_s,formal,repeat_bitwise");
  auto errors = csv(out / "pair_errors.csv",
                    "pair,formal,relative_l2,max_over_peak,max_absolute,changed_complex_words");
  auto metadata = csv(out / "metadata.csv",
                      "frequency_hz,nside,pixels,day,seed,t0,tlen,half_baselines,blocks,tile,orbit_init_s");
  metadata << "10000000," << kNside << ',' << kPixels << ',' << kDay << ',' << kSeed
           << ',' << kStart << ',' << kTimes << ',' << kHalf << ',' << kBlocks << ','
           << kTile << ',' << orbit_init_s << std::endl;
  std::array<std::vector<float2>,2> current{std::vector<float2>(kHalf),std::vector<float2>(kHalf)};
  std::array<std::vector<float2>,2> frozen;
  bool passed = true;
  for (int pair = 0; pair <= kPairs; ++pair) {
    const bool fast_first = pair == 0 || pair == 1 || pair == 4;
    const char* order = fast_first ? "AB" : "BA";
    for (int i = 0; i < 2; ++i) {
      const bool fast = (i == 0) == fast_first;
      const int branch = fast ? 0 : 1;
      const double seconds = launch(fast);
      CHECK_CUDA(cudaMemcpy(current[branch].data(), V, kHalf*sizeof(float2), cudaMemcpyDeviceToHost));
      int same = -1;
      if (pair == 1) frozen[branch] = current[branch];
      if (pair >= 1)
        same = std::memcmp(frozen[branch].data(), current[branch].data(), kHalf*sizeof(float2)) == 0;
      timing << pair << ',' << order << ',' << i+1 << ','
             << (fast ? "tilecone" : "point_only") << ',' << seconds << ',' << (pair > 0)
             << ',' << same << std::endl;
      if (pair >= 1 && !same) passed = false;
    }
    const auto e = compare(current[0], current[1]);
    errors << pair << ',' << (pair > 0) << ',' << e.relative_l2 << ',' << e.max_over_peak
           << ',' << e.max_abs << ',' << e.changed << std::endl;
    if (e.relative_l2 > kTileTolerance || e.max_over_peak > kTileTolerance) passed = false;
    std::cout << "pair=" << pair << " order=" << order << " relative_L2=" << e.relative_l2
              << " changed=" << e.changed << std::endl;
    if (pair == 1) {
      write_array(out / "tilecone_first_formal.f32c", current[0]);
      write_array(out / "point_only_first_formal.f32c", current[1]);
    }
  }

  CHECK_CUDA(cudaFreeHost(staging));
  orbit_gen_destroy(orbit);
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(finish));
  CHECK_CUDA(cudaStreamDestroy(stream));
  if (!passed) {
    std::cerr << "Forward numerical or repeatability gate failed. Raw timings and outputs retained.\n";
    return 2;
  }
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
