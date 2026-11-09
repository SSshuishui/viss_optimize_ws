// Daily full-field raw N/W export. The weighted CUDA kernels are the exact
// archived T01 corrected-DCF diagnostic headers.
#include "weighted_diagnostic/common.hpp"
#include "weighted_diagnostic/orbit_online.cuh"
#include "weighted_diagnostic/viss_recon_kernel.cuh"

#include <openssl/sha.h>
#include <array>
#include <chrono>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
constexpr int kNside = 4096;
constexpr int kPixels = 12 * kNside * kNside;
constexpr int kSegments = 10;
constexpr int kSamplesPerSegment = 6981;
constexpr int kHalfPerSegment = 28 * kSamplesPerSegment;
constexpr int kTile = 256;
constexpr int kTiles = (kPixels + kTile - 1) / kTile;
constexpr size_t kOutputChunk = 1 << 20;
constexpr char kDcfSha[] = "8f5d7f206e5dc1d51282b5be43bfbc237b61d7d63a4665ee10997a244c7e4d2f";

struct CacheHeader {
  char magic[8];
  int32_t nside, day, segment, t0, tlen, half_baselines, gpu_count;
};
struct CacheRow {
  float u, v, w, x1, y1, z1, inv1, x2, y2, z2, inv2, raw_re, raw_im, re, im;
};
static_assert(sizeof(CacheHeader) == 36 && sizeof(CacheRow) == 60, "PREPLAY1 cache ABI");
static_assert(sizeof(float) == 4 && sizeof(double) == 8, "output ABI");

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

double event_seconds(cudaEvent_t start, cudaEvent_t finish) {
  float milliseconds = 0;
  CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, finish));
  return milliseconds * .001;
}

class Digest {
  SHA256_CTX ctx_;
public:
  Digest() { if (SHA256_Init(&ctx_) != 1) throw std::runtime_error("SHA256_Init"); }
  void add(const void* bytes, size_t n) {
    if (SHA256_Update(&ctx_, bytes, n) != 1) throw std::runtime_error("SHA256_Update");
  }
  std::string finish() {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    if (SHA256_Final(hash, &ctx_) != 1) throw std::runtime_error("SHA256_Final");
    std::ostringstream out;
    for (auto byte : hash) out << std::hex << std::setfill('0') << std::setw(2) << int(byte);
    return out.str();
  }
};

std::string hash_file(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open " + path.string());
  Digest hash;
  std::array<char,65536> buf;
  while (in.read(buf.data(), buf.size()) || in.gcount()) hash.add(buf.data(), in.gcount());
  if (!in.eof() || in.bad()) throw std::runtime_error("Cannot hash " + path.string());
  return hash.finish();
}

class Allocations {
  std::vector<void*> device_;
  std::vector<void*> pinned_;
public:
  template<class T> T* device(size_t n) {
    T* ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&ptr, n * sizeof(T)));
    device_.push_back(ptr);
    return ptr;
  }
  template<class T> T* pinned(size_t n) {
    T* ptr = nullptr;
    CHECK_CUDA(cudaMallocHost(&ptr, n * sizeof(T)));
    pinned_.push_back(ptr);
    return ptr;
  }
  ~Allocations() {
    for (auto ptr : pinned_) cudaFreeHost(ptr);
    for (auto ptr : device_) cudaFree(ptr);
  }
};

struct HostSegment {
  CacheHeader header{};
  std::array<std::vector<float>,9> geometry;
  std::vector<float2> visibility;
  std::string name, sha256;
  uintmax_t bytes = 0;
};

HostSegment load_segment(const fs::path& cache, int day, int segment) {
  HostSegment result;
  result.name = "day" + std::to_string(day) + "_seg" + std::to_string(segment) + ".bin";
  const auto path = cache / result.name;
  const auto expected_size = sizeof(CacheHeader) + size_t(kHalfPerSegment) * sizeof(CacheRow);
  result.bytes = fs::file_size(path);
  if (result.bytes != expected_size) throw std::runtime_error("PREPLAY1 size mismatch: " + path.string());
  std::ifstream in(path, std::ios::binary);
  auto& h = result.header;
  in.read(reinterpret_cast<char*>(&h), sizeof(h));
  if (!in || std::memcmp(h.magic, "PREPLAY1", 8) || h.nside != kNside || h.day != day ||
      h.segment != segment || h.half_baselines != kHalfPerSegment || h.gpu_count != 1 ||
      h.tlen != kSamplesPerSegment || h.t0 != (segment - 1) * kSamplesPerSegment)
    throw std::runtime_error("PREPLAY1 header mismatch: " + path.string());
  std::vector<CacheRow> rows(kHalfPerSegment);
  in.read(reinterpret_cast<char*>(rows.data()), rows.size() * sizeof(CacheRow));
  if (!in || in.peek() != EOF || in.bad()) throw std::runtime_error("PREPLAY1 body mismatch: " + path.string());
  Digest hash;
  hash.add(&h, sizeof(h));
  hash.add(rows.data(), rows.size() * sizeof(CacheRow));
  result.sha256 = hash.finish();

  for (auto& a : result.geometry) a.resize(kHalfPerSegment);
  result.visibility.resize(kHalfPerSegment);
  for (int i = 0; i < kHalfPerSegment; ++i) {
    const auto& row = rows[i];
    const float vals[] = {row.u, row.v, row.w, row.x1, row.y1, row.z1, row.x2, row.y2, row.z2};
    for (int j = 0; j < 9; ++j) {
      if (!std::isfinite(vals[j])) throw std::runtime_error("Nonfinite cached geometry: " + path.string());
      result.geometry[j][i] = vals[j];
    }
    if (!std::isfinite(row.re) || !std::isfinite(row.im))
      throw std::runtime_error("Nonfinite cached consumed visibility: " + path.string());
    result.visibility[i] = make_float2(row.re, row.im);
  }
  return result;
}

struct OutputStats {
  Digest hash;
  size_t count = 0, zeros = 0;
  double min = std::numeric_limits<double>::infinity();
  double max = -std::numeric_limits<double>::infinity();

  template<class T> void append(const T* data, size_t n, bool weight) {
    hash.add(data, n * sizeof(T));
    for (size_t j = 0; j < n; ++j) {
      double v = data[j];
      if (!std::isfinite(v) || (weight && v < 0))
        throw std::runtime_error(weight ? "Invalid raw W" : "Invalid raw N");
      min = std::min(min, v);
      max = std::max(max, v);
      zeros += v == 0;
    }
    count += n;
  }

  void json(std::ostream& out, const char* file, const char* dtype, const char* format, size_t word) {
    out << "{\"file\":" << std::quoted(file) << ",\"dtype\":" << std::quoted(dtype)
        << ",\"format\":" << std::quoted(format) << ",\"bytes\":" << count * word
        << ",\"count\":" << count << ",\"finite_count\":" << count << ",\"nonfinite\":0"
        << ",\"zeros\":" << zeros << ",\"min\":" << min << ",\"max\":" << max
        << ",\"sha256\":" << std::quoted(hash.finish()) << '}';
  }
};

std::ofstream open_partial(const fs::path& out, const char* name,
                           std::ios::openmode mode = std::ios::out) {
  const auto file = out / (std::string(name) + ".part");
  std::ofstream stream(file, mode);
  if (!stream) throw std::runtime_error("Cannot write " + file.string());
  stream << std::setprecision(17);
  return stream;
}

void close_checked(std::ofstream& stream, const char* label) {
  stream.close();
  if (!stream) throw std::runtime_error(std::string("Failed to write ") + label);
}

int parse_integer(const std::string& value, const char* option) {
  size_t consumed = 0;
  int result = std::stoi(value, &consumed);
  if (consumed != value.size()) throw std::runtime_error(std::string("Invalid ") + option);
  return result;
}

int main(int argc, char** argv) try {
  const int day = parse_integer(get_arg(argc, argv, "--day", "0"), "--day");
  const int nside = parse_integer(get_arg(argc, argv, "--nside", "4096"), "--nside");
  const int nseg = parse_integer(get_arg(argc, argv, "--max_segments", "10"), "--max_segments");
  const std::string gpu = get_arg(argc, argv, "--gpus", "");
  const fs::path cache = get_arg(argc, argv, "--cache_root", "");
  const fs::path dcf_path = get_arg(argc, argv, "--dcf_bin", "");
  const fs::path out = get_arg(argc, argv, "--out_dir", "");
  if (day < 1 || day > 450 || nside != kNside || nseg != kSegments ||
      gpu.find(',') != std::string::npos || !fs::is_directory(cache) ||
      !fs::is_regular_file(dcf_path) || !fs::is_directory(out))
    throw std::runtime_error("Require day 1..450, Nside4096, ten full original segments, "
                             "one visible GPU, valid --cache_root, --dcf_bin and --out_dir.");
  const char* outputs[] = {"raw_N.f32", "W.f64", "segments.csv", "full_stats.json", "metrics.json"};
  for (const auto* name : outputs) {
    if (fs::symlink_status(out / name).type() != fs::file_type::not_found ||
        fs::symlink_status(out / (std::string(name) + ".part")).type() != fs::file_type::not_found)
      throw std::runtime_error("Refusing to replace an existing output in " + out.string());
  }

  const auto cache_start = Clock::now();
  DcfMbBinPayload payload;
  const auto dcf_sha = hash_file(dcf_path);
  if (dcf_sha != kDcfSha || !load_dcf_mb_bin(dcf_path.string(), payload) ||
      payload.btag != "10M" || payload.dcf_days != 450 || payload.orbit_seed != 42)
    throw std::runtime_error("DCF table differs from the frozen corrected-DCF identity");
  std::vector<HostSegment> host_segments;
  host_segments.reserve(kSegments);
  for (int segment = 1; segment <= kSegments; ++segment)
    host_segments.push_back(load_segment(cache, day, segment));
  const double cache_s = seconds_since(cache_start);

  if (!gpu.empty()) {
    setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 1);
    setenv("CUDA_VISIBLE_DEVICES", gpu.c_str(), 1);
  }
  int device_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (device_count != 1) throw std::runtime_error("Require exactly one visible CUDA GPU");
  CHECK_CUDA(cudaSetDevice(0));
  cudaDeviceProp properties{};
  CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));

  const auto init_start = Clock::now();
  size_t free_before = 0, total = 0;
  CHECK_CUDA(cudaMemGetInfo(&free_before, &total));
  const size_t estimated_min = size_t(kPixels) * 24 + size_t(kTiles) * 20 +
                               size_t(kHalfPerSegment) * 48 + 256 * 1024 * 1024ULL;
  if (free_before < estimated_min + 512 * 1024 * 1024ULL)
    throw std::runtime_error("Insufficient free GPU memory for the complete raw N/W output");

  Allocations mem;
  float* l = mem.device<float>(kPixels);
  float* m = mem.device<float>(kPixels);
  float* n = mem.device<float>(kPixels);
  float* rawN = mem.device<float>(kPixels);
  double* rawW = mem.device<double>(kPixels);
  std::array<float*,5> cone;
  for (auto& p : cone) p = mem.device<float>(kTiles);
  std::array<float*,9> d;
  for (auto& p : d) p = mem.device<float>(kHalfPerSegment);
  float2* visibility = mem.device<float2>(kHalfPerSegment);
  float* pair_weight = mem.device<float>(kHalfPerSegment);
  float* dcf = mem.device<float>(payload.dcf.size());
  CHECK_CUDA(cudaMemcpy(dcf, payload.dcf.data(), payload.dcf.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  cudaStream_t compute, xfer, reduce;
  CHECK_CUDA(cudaStreamCreate(&compute));
  CHECK_CUDA(cudaStreamCreate(&xfer));
  CHECK_CUDA(cudaStreamCreate(&reduce));
  pix2lmn_nest_kernel<<<kTiles,kTile,0,compute>>>(kNside, 0, kPixels, l, m, n);
  launch_build_tile_cone_meta_runtime(kTile, compute, l, m, n, kPixels,
                                    cone[0], cone[1], cone[2], cone[3], cone[4], kTiles);
  CHECK_CUDA(cudaPeekAtLastError());
  CHECK_CUDA(cudaMemsetAsync(rawN, 0, size_t(kPixels) * sizeof(float), compute));
  CHECK_CUDA(cudaMemsetAsync(rawW, 0, size_t(kPixels) * sizeof(double), compute));

  GpuCtx ctx;
  ctx.dev = 0;
  ctx.compute_stream = compute;
  ctx.reduce_stream = reduce;
  ctx.recon_n_chunk = kPixels;
  ctx.ntile_recon = kTiles;
  ctx.d_l_recon = l;
  ctx.d_m_recon = m;
  ctx.d_n_recon = n;
  ctx.d_Cacc = rawN;
  ctx.d_diag_weighted_W = rawW;
  ctx.d_diag_pairw = pair_weight;
  ctx.d_tile_cx_recon = cone[0];
  ctx.d_tile_cy_recon = cone[1];
  ctx.d_tile_cz_recon = cone[2];
  ctx.d_tile_cosA_recon = cone[3];
  ctx.d_tile_sinA_recon = cone[4];

  GpuSegSlot slot;
  slot.d_u = d[0];
  slot.d_v = d[1];
  slot.d_w = d[2];
  slot.d_x1 = d[3];
  slot.d_y1 = d[4];
  slot.d_z1 = d[5];
  slot.d_x2 = d[6];
  slot.d_y2 = d[7];
  slot.d_z2 = d[8];
  slot.d_Viss = visibility;
  CHECK_CUDA(cudaEventCreateWithFlags(&slot.pairw_ready, cudaEventDisableTiming));

  constexpr int max_tasks = kTiles * ((OP_RECON_PLAN_BL_CHUNK + OP_RECON_TASK_BL - 1) /
                                    OP_RECON_TASK_BL);
  cub::CountingInputIterator<int> counting(0);
  size_t plan_bytes = 0;
  for (auto& plan : ctx.recon_plan) {
    plan.d_vv_flags = mem.device<unsigned char>(max_tasks);
    plan.d_mixed_flags = mem.device<unsigned char>(max_tasks);
    plan.d_vv_tasks = mem.device<int>(max_tasks);
    plan.d_mixed_tasks = mem.device<int>(max_tasks);
    plan.d_num_vv = mem.device<int>(1);
    plan.d_num_mixed = mem.device<int>(1);
    plan.h_num_vv = mem.pinned<int>(1);
    plan.h_num_mixed = mem.pinned<int>(1);
    CHECK_CUDA(cub::DeviceSelect::Flagged(nullptr, plan.select_temp_vv_bytes, counting,
        plan.d_vv_flags, plan.d_vv_tasks, plan.d_num_vv, max_tasks, reduce));
    CHECK_CUDA(cub::DeviceSelect::Flagged(nullptr, plan.select_temp_mixed_bytes, counting,
        plan.d_mixed_flags, plan.d_mixed_tasks, plan.d_num_mixed, max_tasks, reduce));
    plan.d_select_temp_vv = mem.device<unsigned char>(plan.select_temp_vv_bytes);
    plan.d_select_temp_mixed = mem.device<unsigned char>(plan.select_temp_mixed_bytes);
    CHECK_CUDA(cudaEventCreate(&plan.ready));
    CHECK_CUDA(cudaEventCreateWithFlags(&plan.compute_done, cudaEventDisableTiming));
    CHECK_CUDA(cudaEventRecord(plan.compute_done, compute));
    plan_bytes += size_t(max_tasks) * 10 + 8 + plan.select_temp_vv_bytes + plan.select_temp_mixed_bytes;
  }

  cudaEvent_t h2d_start, h2d_end, prep_end, recon_start, recon_end;
  for (auto event : {&h2d_start, &h2d_end, &prep_end, &recon_start, &recon_end})
    CHECK_CUDA(cudaEventCreate(event));
  CHECK_CUDA(cudaDeviceSynchronize());
  size_t free_init = 0;
  CHECK_CUDA(cudaMemGetInfo(&free_init, &total));
  size_t min_free = free_init;
  const double init_s = seconds_since(init_start);

  std::cout << "DIRECT_DAILY day=" << day << " GPU=" << properties.name << " pixels=" << kPixels
            << " cache_decode_s=" << cache_s << " init_s=" << init_s << std::endl;
  auto segment_csv = open_partial(out, "segments.csv");
  segment_csv << "segment,t0,tlen,half_baselines,h2d_event_s,prep_wall_s,recon_wall_s,"
                 "recon_stream_span_s,segment_wall_s,vv_tasks,mixed_tasks,cache_sha256\n";
  const float R = 1737.1e3f, h = 300e3f;
  const float cosphi = cosf(float(M_PI) - asinf(R / (R + h)));
  constexpr int bl_grid = (kHalfPerSegment + kTile - 1) / kTile;
  const auto loop_start = Clock::now();
  for (const auto& seg : host_segments) {
    const auto segment_start = Clock::now();
    CHECK_CUDA(cudaEventRecord(h2d_start, xfer));
    for (int j = 0; j < 9; ++j)
      CHECK_CUDA(cudaMemcpyAsync(d[j], seg.geometry[j].data(), kHalfPerSegment * sizeof(float),
                                 cudaMemcpyHostToDevice, xfer));
    CHECK_CUDA(cudaMemcpyAsync(visibility, seg.visibility.data(), kHalfPerSegment * sizeof(float2),
                               cudaMemcpyHostToDevice, xfer));
    CHECK_CUDA(cudaEventRecord(h2d_end, xfer));
    CHECK_CUDA(cudaEventSynchronize(h2d_end));

    const auto prep_start = Clock::now();
    normalize3_inplace_kernel<<<bl_grid,kTile,0,compute>>>(d[3], d[4], d[5], kHalfPerSegment);
    normalize3_inplace_kernel<<<bl_grid,kTile,0,compute>>>(d[6], d[7], d[8], kHalfPerSegment);
    compute_pair_weight_half_kernel<<<bl_grid,kTile,0,compute>>>(
        d[0], d[1], d[2], kHalfPerSegment, dcf, payload.dcf.size(), pair_weight);
    apply_pair_weight_to_viss_half_kernel<<<bl_grid,kTile,0,compute>>>(
        visibility, pair_weight, kHalfPerSegment);
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaEventRecord(slot.pairw_ready, compute));
    CHECK_CUDA(cudaEventRecord(prep_end, compute));
    CHECK_CUDA(cudaEventSynchronize(prep_end));
    const double prep_s = seconds_since(prep_start);

    const auto recon_wall_start = Clock::now();
    CHECK_CUDA(cudaEventRecord(recon_start, compute));
    const auto stats = run_recon_planaware_segment(ctx, slot, kHalfPerSegment, cosphi);
    CHECK_CUDA(cudaEventRecord(recon_end, compute));
    CHECK_CUDA(cudaDeviceSynchronize());
    const double recon_s = seconds_since(recon_wall_start);
    size_t free_now = 0;
    CHECK_CUDA(cudaMemGetInfo(&free_now, &total));
    min_free = std::min(min_free, free_now);
    segment_csv << seg.header.segment << ',' << seg.header.t0 << ',' << seg.header.tlen << ','
                << seg.header.half_baselines << ',' << event_seconds(h2d_start, h2d_end) << ','
                << prep_s << ',' << recon_s << ',' << event_seconds(recon_start, recon_end) << ','
                << seconds_since(segment_start) << ',' << stats.vv_tasks << ',' << stats.mixed_tasks
                << ',' << seg.sha256 << std::endl;
    if (!segment_csv) throw std::runtime_error("Failed to record segment");
    std::cout << "INVERSE_SEG_DONE day=" << day << " segment=" << seg.header.segment
              << '/' << kSegments << " recon_wall_s=" << recon_s << std::endl;
  }
  const double loop_s = seconds_since(loop_start);
  close_checked(segment_csv, "segments.csv");

  const auto output_start = Clock::now();
  OutputStats n_stats, w_stats;
  std::vector<float> hostN(kOutputChunk);
  std::vector<double> hostW(kOutputChunk);
  auto fileN = open_partial(out, "raw_N.f32", std::ios::binary | std::ios::out);
  auto fileW = open_partial(out, "W.f64", std::ios::binary | std::ios::out);
  for (size_t begin = 0; begin < size_t(kPixels); begin += kOutputChunk) {
    const size_t count = std::min(kOutputChunk, size_t(kPixels) - begin);
    CHECK_CUDA(cudaMemcpy(hostN.data(), rawN + begin, count * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hostW.data(), rawW + begin, count * sizeof(double), cudaMemcpyDeviceToHost));
    n_stats.append(hostN.data(), count, false);
    w_stats.append(hostW.data(), count, true);
    fileN.write(reinterpret_cast<const char*>(hostN.data()), count * sizeof(float));
    fileW.write(reinterpret_cast<const char*>(hostW.data()), count * sizeof(double));
    if (!fileN || !fileW) throw std::runtime_error("Failed to write raw output chunks");
  }
  close_checked(fileN, "raw_N.f32");
  close_checked(fileW, "W.f64");
  if (fs::file_size(out / "raw_N.f32.part") != size_t(kPixels) * sizeof(float) ||
      fs::file_size(out / "W.f64.part") != size_t(kPixels) * sizeof(double))
    throw std::runtime_error("Raw output size mismatch");
  const double output_s = seconds_since(output_start);

  auto full_stats = open_partial(out, "full_stats.json");
  full_stats << "{\"rawN\":";
  n_stats.json(full_stats, "raw_N.f32", "float32", "<f4", sizeof(float));
  full_stats << ",\"W\":";
  w_stats.json(full_stats, "W.f64", "float64", "<f8", sizeof(double));
  full_stats << "}\n";
  close_checked(full_stats, "full_stats.json");

  auto result = open_partial(out, "metrics.json");
  result << "{\n  \"state\":\"completed\",\n  \"method\":\"full_corrected_dcf_weighted_reference\",\n"
         << "  \"day\":" << day << ",\"nside\":" << kNside << ",\"segments\":" << kSegments
         << ",\"half_baselines\":" << kSegments * kHalfPerSegment << ",\"output_pixels\":" << kPixels
         << ",\"ordering\":\"NEST\",\n  \"forward_executed\":false,"
         << "\"forward_gpu_count\":1,\"dcf_sha256\":" << std::quoted(dcf_sha)
         << ",\"dcf_days\":450,\"orbit_seed\":42,\n  \"gpu\":" << std::quoted(properties.name)
         << ",\"compute_capability\":" << std::quoted(std::to_string(properties.major) + "." +
                                                       std::to_string(properties.minor))
         << ",\n  \"cache_load_decode_wall_s\":" << cache_s << ",\"init_wall_s\":" << init_s
         << ",\"inverse_loop_wall_s\":" << loop_s << ",\"raw_export_wall_s\":" << output_s
         << ",\"estimated_min_allocation_bytes\":" << estimated_min
         << ",\"plan_allocated_bytes\":" << plan_bytes
         << ",\"observed_init_device_occupied_bytes\":" << total - free_init
         << ",\"observed_boundary_max_device_occupied_bytes\":" << total - min_free << ",\n"
         << "  \"normalization\":\"none; export FP32 raw numerator N and FP64 weighted denominator W\",\n"
         << "  \"visibility\":\"PREPLAY1 G1 consumed-n V, with no additional phase correction\",\n"
         << "  \"caches\":[";
  for (size_t i = 0; i < host_segments.size(); ++i) {
    const auto& seg = host_segments[i];
    if (i) result << ',';
    result << "\n    {\"name\":" << std::quoted(seg.name) << ",\"sha256\":" << std::quoted(seg.sha256)
           << ",\"bytes\":" << seg.bytes << ",\"segment\":" << seg.header.segment << '}';
  }
  result << "\n  ]\n}\n";
  close_checked(result, "metrics.json");

  for (auto event : {h2d_start, h2d_end, prep_end, recon_start, recon_end, slot.pairw_ready})
    CHECK_CUDA(cudaEventDestroy(event));
  for (auto& plan : ctx.recon_plan) {
    CHECK_CUDA(cudaEventDestroy(plan.ready));
    CHECK_CUDA(cudaEventDestroy(plan.compute_done));
  }
  for (auto stream : {compute, xfer, reduce}) CHECK_CUDA(cudaStreamDestroy(stream));
  // metrics.json is renamed last. The driver must ignore a partial attempt
  // without this marker even if one or more large files already have final names.
  for (const auto* name : outputs) fs::rename(out / (std::string(name) + ".part"), out / name);
  std::cout << "DIRECT_DAILY_DONE day=" << day << " inverse_loop_wall_s=" << loop_s
            << " raw_export_wall_s=" << output_s << std::endl;
  return 0;
} catch (const std::exception& error) {
  std::cerr << "ERROR " << error.what() << '\n';
  return 2;
}
