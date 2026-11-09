// Isolated MPI transport for the archived paper STS operator.
// The numerical kernels below come from unchanged vendor/final_sts headers.
#include <mpi.h>
#include "common_mirror_tile_vissfast.hpp"
#include <array>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <type_traits>
#if defined(__linux__)
#include <sys/resource.h>
#endif

namespace fs = std::filesystem;
static int world_rank = -1;
[[noreturn]] static void fatal(const std::string& message) {
  std::cerr << "[rank " << world_rank << "] ERROR: " << message << std::endl;
  int initialized = 0, finalized = 0;
  MPI_Initialized(&initialized);
  if (initialized) MPI_Finalized(&finalized);
  if (initialized && !finalized) MPI_Abort(MPI_COMM_WORLD, 1);
  std::exit(1);
}
static void mpi_check(int code, const char* expression, int line) {
  if (code == MPI_SUCCESS) return;
  char error[MPI_MAX_ERROR_STRING]; int length = 0;
  MPI_Error_string(code, error, &length);
  fatal(std::string("MPI at line ") + std::to_string(line) + ": " + expression + ": " + std::string(error, length));
}
#define MC(call) mpi_check((call), #call, __LINE__)
static void cuda_check(cudaError_t code, const char* expression, int line) {
  if (code != cudaSuccess) fatal(std::string("CUDA at line ") + std::to_string(line) + ": " + expression + ": " + cudaGetErrorString(code));
}
// Redefine before including any host implementations: a failed rank must abort
// its peers instead of leaving them blocked in a collective.
#undef CHECK_CUDA
#define CHECK_CUDA(call) cuda_check((call), #call, __LINE__)
#include "orbit_online_mirror_tile_vissfast.cuh"
#include "viss_recon_kernel_mirror_tile_vissfast.cuh"

static_assert(sizeof(float) == 4 && sizeof(float2) == 8 && sizeof(uint32_t) == 4, "binary output types");
static constexpr int TILE_PIX = 256;
static constexpr long long VALIDATION_MAX_PIXELS = 49152;

static std::string quote(const std::string& value) {
  std::ostringstream out; out << '"';
  for (unsigned char c : value) {
    if (c == '"') out << "\\\"";
    else if (c == '\\') out << "\\\\";
    else if (c == '\n') out << "\\n";
    else if (c == '\r') out << "\\r";
    else if (c == '\t') out << "\\t";
    else if (c < 32) { out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (unsigned)c << std::dec; }
    else out << c;
  }
  out << '"'; return out.str();
}
// Bounded observations at stage boundaries, not resource polling. Device-wide
// usage includes CUDA contexts, displays and other processes. Allocations that
// are created and freed inside a stage can exceed every observed value here.
struct MemoryStageObservation {
  uint64_t samples=0,min_free_bytes=UINT64_MAX,max_used_bytes=0;
  int day_at_max=-1,segment_at_max=-1;
  double first_elapsed_s=0,last_elapsed_s=0;
};
struct DeviceMemoryObservations {
  uint64_t initial_free_bytes,initial_total_bytes;
  double initialized_at=MPI_Wtime();
  std::map<std::string,MemoryStageObservation> stages;
  DeviceMemoryObservations(size_t free_bytes,size_t total_bytes)
      :initial_free_bytes(free_bytes),initial_total_bytes(total_bytes){}
  void capture(const std::string& label,int day=-1,int segment=-1) {
    size_t free_bytes=0,total_bytes=0;CHECK_CUDA(cudaMemGetInfo(&free_bytes,&total_bytes));
    auto& row=stages[label];double elapsed=MPI_Wtime()-initialized_at;
    if(!row.samples)row.first_elapsed_s=elapsed;
    uint64_t used=(uint64_t)total_bytes-(uint64_t)free_bytes;
    if(!row.samples||used>row.max_used_bytes){row.max_used_bytes=used;row.day_at_max=day;row.segment_at_max=segment;}
    row.min_free_bytes=std::min(row.min_free_bytes,(uint64_t)free_bytes);
    row.last_elapsed_s=elapsed;++row.samples;
  }
  std::string json() const {
    uint64_t observed_max=initial_total_bytes-initial_free_bytes;
    for(const auto& item:stages)observed_max=std::max(observed_max,item.second.max_used_bytes);
    std::ostringstream out;out<<std::setprecision(17)
      <<"{\"schema\":\"cuda_stage_boundary_memory_v1\",\"measurement\":\"cudaMemGetInfo_device_total_minus_free\",\"continuous_monitoring\":false,\"absolute_peak_measured\":false,\"process_peak_measured\":false,\"initial_free_device_bytes\":"<<initial_free_bytes
      <<",\"initial_device_total_bytes\":"<<initial_total_bytes<<",\"observed_device_used_bytes_high_water\":"<<observed_max
      <<",\"observed_device_used_delta_from_initial_bytes\":"<<(int64_t(observed_max)-int64_t(initial_total_bytes-initial_free_bytes))
      <<",\"qualification\":\"Stage-boundary samples only; includes contexts/displays/other processes; misses temporary allocations freed within stages; serial-check samples include extra validation replay arrays\",\"stages\":[";
    bool first=true;for(const auto& item:stages){if(!first)out<<',';first=false;const auto& r=item.second;
      out<<"{\"label\":"<<quote(item.first)<<",\"sample_count\":"<<r.samples<<",\"minimum_observed_free_bytes\":"<<r.min_free_bytes
        <<",\"maximum_observed_device_used_bytes\":"<<r.max_used_bytes<<",\"day_at_maximum\":"<<r.day_at_max
        <<",\"segment_at_maximum\":"<<r.segment_at_max<<",\"first_elapsed_s\":"<<r.first_elapsed_s<<",\"last_elapsed_s\":"<<r.last_elapsed_s<<'}';}
    out<<"]}";return out.str();
  }
};
static std::string padded(int n, int width = 4) { std::ostringstream s; s << std::setw(width) << std::setfill('0') << n; return s.str(); }
static uint64_t fnv_update(uint64_t h, const void* data, size_t bytes) {
  const auto* p = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < bytes; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
  return h;
}
static uint64_t string_fingerprint(const std::string& s) { return fnv_update(14695981039346656037ULL, s.data(), s.size()); }
static std::string hex64(uint64_t n) { std::ostringstream s; s << std::hex << std::setw(16) << std::setfill('0') << n; return s.str(); }

struct Config {
  int nside = 64, day_start = 1, day_count = 1, segs = 2;
  int time_start = 0, time_count = 16, tlen_limit = 0, segments_limit = 0;
  float grid_cell_scale = 1.0f;
  uint64_t orbit_seed = 42;
  int validation_node_size = 0;
  std::string btag = "10M", input, output_dir = "cluster_output", recon_mode = "rep", strategy = "flat";
  bool synthetic = true, allow_shared_gpu = false, serial_check = false, dump_validation = false, help = false;
  bool write_images = true;
};
static void usage() {
  std::cout << "MPI STS: one rank per GPU, CPU-buffer collectives, unchanged archived kernels.\n"
    << "--nside=64 --synthetic=1 [--input=SKY_F32_BIN --synthetic=0]\n"
    << "--day_start=1 --day_count=1 --segs=2 --time_start=0 --time_count=16\n"
    << "--segments-limit=0 --tlen-limit=0 --grid_cell_scale=1 --orbit_seed=42\n"
    << "--btag=10M --recon-mode=rep|exact --output-dir=NEW_DIRECTORY\n"
    << "--allow-shared-gpu=0 --serial-check=0 --dump-validation=0\n"
    << "--write-images=1 (0 suppresses image .bin output only; downloads/gather/normalization remain)\n"
    << "--strategy=flat|hierarchical --validation-node-size=0\n"
    << "Default is a bounded synthetic test. --time_count=0 explicitly selects the full day.\n"
    << "Boolean flags accept =0/=1 or a bare flag. Underscores and hyphens are equivalent.\n"
    << "Only NEST power-of-two nside>=8; 256-pixel tile ownership; whole-map MPI Gatherv is INT_MAX limited.\n"
    << "--serial-check/--dump-validation are restricted to <=49152 pixels.\n"
    << "--validation-node-size=N simulates communication groups (partial tail allowed),\n"
    << "requires --allow-shared-gpu=1 and <=49152 pixels; it never simulates physical hosts.\n";
}
static Config parse_config(int argc, char** argv) {
  std::map<std::string, std::string> opts;
  const std::set<std::string> allowed = {"nside","day","day-start","day-count","segs","time-start","time-count","tlen-limit","segments-limit","grid-cell-scale","orbit-seed","btag","input","output-dir","out-dir","recon-mode","strategy","validation-node-size","synthetic","allow-shared-gpu","serial-check","dump-validation","write-images","help"};
  const std::set<std::string> bools = {"synthetic","allow-shared-gpu","serial-check","dump-validation","write-images","help"};
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--", 0) != 0) throw std::runtime_error("arguments must start with --: " + arg);
    auto pos = arg.find('='); std::string key = arg.substr(2, pos == std::string::npos ? pos : pos - 2);
    std::replace(key.begin(), key.end(), '_', '-');
    if (!allowed.count(key)) throw std::runtime_error("unknown argument: --" + key);
    if (opts.count(key)) throw std::runtime_error("duplicate argument: --" + key);
    std::string value;
    if (pos != std::string::npos) value = arg.substr(pos + 1);
    else if (bools.count(key)) value = "1";
    else { if (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0) throw std::runtime_error("missing value: --" + key); value = argv[++i]; }
    opts[key] = value;
  }
  auto integer = [&](const char* key, long long def) { if (!opts.count(key)) return def; size_t end = 0; long long x = std::stoll(opts.at(key), &end); if (end != opts.at(key).size()) throw std::runtime_error(std::string("invalid integer: ") + key); return x; };
  auto i32 = [&](const char* key, int def) { long long x = integer(key, def); if (x < INT_MIN || x > INT_MAX) throw std::runtime_error(std::string("integer range: ") + key); return (int)x; };
  auto boolean = [&](const char* key, bool def) { if (!opts.count(key)) return def; auto s = opts.at(key); if (s == "1" || s == "true") return true; if (s == "0" || s == "false") return false; throw std::runtime_error(std::string("boolean must be 0/1: ") + key); };
  auto str = [&](const char* key, const std::string& def) { return opts.count(key) ? opts.at(key) : def; };
  Config c;
  c.help = boolean("help", false); if (c.help) return c;
  c.nside = i32("nside", c.nside); c.day_start = i32("day-start", i32("day", c.day_start)); c.day_count = i32("day-count", c.day_count);
  c.segs = i32("segs", c.segs); c.time_start = i32("time-start", c.time_start); c.time_count = i32("time-count", c.time_count);
  c.tlen_limit = i32("tlen-limit", 0); c.segments_limit = i32("segments-limit", 0);
  c.validation_node_size = i32("validation-node-size", 0); c.strategy = str("strategy", "flat");
  c.btag = str("btag", c.btag); c.input = str("input", ""); c.output_dir = str("output-dir", str("out-dir", c.output_dir)); c.recon_mode = str("recon-mode", "rep");
  c.synthetic = boolean("synthetic", c.input.empty()); c.allow_shared_gpu = boolean("allow-shared-gpu", false); c.serial_check = boolean("serial-check", false); c.dump_validation = boolean("dump-validation", false);
  c.write_images = boolean("write-images", true);
  if (opts.count("grid-cell-scale")) { size_t end = 0; c.grid_cell_scale = std::stof(opts.at("grid-cell-scale"), &end); if (end != opts.at("grid-cell-scale").size()) throw std::runtime_error("invalid grid-cell-scale"); }
  if (opts.count("orbit-seed")) { if (!opts.at("orbit-seed").empty() && opts.at("orbit-seed")[0] == '-') throw std::runtime_error("negative orbit seed"); size_t end = 0; c.orbit_seed = std::stoull(opts.at("orbit-seed"), &end); if (end != opts.at("orbit-seed").size()) throw std::runtime_error("invalid orbit seed"); }
  if (c.nside < 8 || (c.nside & (c.nside - 1))) throw std::runtime_error("nside must be a power of two >=8 (tile-safe reconstruction)");
  if (c.nside > 8192) throw std::runtime_error("cluster prototype restricts nside<=8192: whole-map MPI counts and root output limits");
  if (c.day_start < 1 || c.day_count < 1 || (long long)c.day_start + c.day_count - 1 > 450) throw std::runtime_error("day range must be within1..450");
  if (c.segs < 1 || c.time_start < 0 || c.time_count < 0 || c.tlen_limit < 0 || c.segments_limit < 0) throw std::runtime_error("invalid segment/time limits");
  if (!(c.grid_cell_scale > 0) || !std::isfinite(c.grid_cell_scale)) throw std::runtime_error("grid-cell-scale must be finite and >0");
  if (c.btag != "1M" && c.btag != "10M" && c.btag != "30M") throw std::runtime_error("btag must be1M/10M/30M");
  if (c.recon_mode != "rep" && c.recon_mode != "exact") throw std::runtime_error("recon-mode must be rep/exact");
  if (c.strategy != "flat" && c.strategy != "hierarchical") throw std::runtime_error("strategy must be flat/hierarchical");
  if (c.validation_node_size < 0) throw std::runtime_error("validation-node-size must be >=0");
  if (c.synthetic == !c.input.empty()) throw std::runtime_error("choose synthetic=1 without input OR synthetic=0 with input");
  if (c.output_dir.empty()) throw std::runtime_error("output-dir cannot be empty");
  if (!c.write_images && (c.serial_check || c.dump_validation)) throw std::runtime_error("write-images=0 requires serial-check=0 and dump-validation=0; validation image output is not silently suppressed");
  long long npix = 12LL * c.nside * c.nside;
  if ((c.serial_check || c.dump_validation) && npix > VALIDATION_MAX_PIXELS) throw std::runtime_error("serial-check/dump-validation limit is49152 pixels");
  if (c.validation_node_size && (!c.allow_shared_gpu || npix > VALIDATION_MAX_PIXELS)) throw std::runtime_error("validation-node-size is tiny validation only: require allow-shared-gpu=1 and <=49152 pixels");
  return c;
}
static std::string config_json(const Config& c) {
  std::ostringstream o; o << std::setprecision(17);
  o << "{\"nside\":" << c.nside << ",\"npix\":" << 12LL*c.nside*c.nside << ",\"day_start\":" << c.day_start << ",\"day_count\":" << c.day_count
    << ",\"segs\":" << c.segs << ",\"time_start\":" << c.time_start << ",\"time_count\":" << c.time_count << ",\"tlen_limit\":" << c.tlen_limit << ",\"segments_limit\":" << c.segments_limit
    << ",\"grid_cell_scale\":" << c.grid_cell_scale << ",\"orbit_seed\":" << c.orbit_seed << ",\"btag\":" << quote(c.btag)
    << ",\"lambda_m\":" << (c.btag == "1M" ? 300.0 : c.btag == "10M" ? 30.0 : 10.0)
    << ",\"recon_mode\":" << quote(c.recon_mode) << ",\"sky_kind\":" << quote(c.synthetic ? "deterministic_synthetic" : "float32_binary") << ",\"input\":" << quote(c.input)
    << ",\"strategy\":" << quote(c.strategy) << ",\"validation_node_size\":" << c.validation_node_size
    << ",\"output_dir\":" << quote(c.output_dir) << ",\"tile_pix\":256,\"synthetic\":" << (c.synthetic ? "true" : "false")
    << ",\"allow_shared_gpu\":" << (c.allow_shared_gpu ? "true" : "false") << ",\"serial_check\":" << (c.serial_check ? "true" : "false") << ",\"dump_validation\":" << (c.dump_validation ? "true" : "false")
    << ",\"write_images\":" << (c.write_images ? "true" : "false") << "}";
  return o.str();
}

struct Ownership { int rank; long long pixel_start, pixel_count, tile_start, tile_count; };
static Ownership ownership(long long npix, int rank, int size) {
  long long tiles = npix / TILE_PIX, base = tiles / size, rem = tiles % size;
  long long start = rank * base + std::min<long long>(rank, rem), count = base + (rank < rem);
  return {rank, start*TILE_PIX, count*TILE_PIX, start, count};
}

// Actual physical-node grouping remains separate from the optional tiny logical
// communication groups. GPU selection and duplicate UUID checks use the former.
struct NodeTransport {
  bool hierarchical = false;
  MPI_Comm node_comm = MPI_COMM_NULL, leaders_comm = MPI_COMM_NULL;
  int node_rank = 0, node_size = 0, group_count = 0, actual_group_count = 0;
  std::vector<int> leader_by_world_rank;
  std::vector<std::vector<int>> groups;
} nodes;
static void initialize_node_transport(const Config& c, MPI_Comm actual_comm,
                                      int actual_local_rank, int size) {
  nodes.hierarchical = c.strategy == "hierarchical";
  int actual_leader = actual_local_rank == 0 ? world_rank : -1;
  MC(MPI_Bcast(&actual_leader,1,MPI_INT,0,actual_comm));
  std::vector<int> actual_ids(size);
  MC(MPI_Allgather(&actual_leader,1,MPI_INT,actual_ids.data(),1,MPI_INT,MPI_COMM_WORLD));
  nodes.actual_group_count = (int)std::set<int>(actual_ids.begin(),actual_ids.end()).size();
  if (c.validation_node_size)
    MC(MPI_Comm_split(MPI_COMM_WORLD,world_rank/c.validation_node_size,world_rank,&nodes.node_comm));
  else MC(MPI_Comm_dup(actual_comm,&nodes.node_comm));
  MC(MPI_Comm_rank(nodes.node_comm,&nodes.node_rank));
  MC(MPI_Comm_size(nodes.node_comm,&nodes.node_size));
  int leader = nodes.node_rank == 0 ? world_rank : -1;
  MC(MPI_Bcast(&leader,1,MPI_INT,0,nodes.node_comm));
  nodes.leader_by_world_rank.resize(size);
  MC(MPI_Allgather(&leader,1,MPI_INT,nodes.leader_by_world_rank.data(),1,MPI_INT,MPI_COMM_WORLD));
  std::map<int,std::vector<int>> membership;
  for (int r=0;r<size;++r) membership[nodes.leader_by_world_rank[r]].push_back(r);
  for (const auto& entry:membership) nodes.groups.push_back(entry.second);
  nodes.group_count=(int)nodes.groups.size();
  // World rank0 is the lowest rank in its node and hence leader communicator0.
  // Leaders are ordered by world rank, matching the map order above.
  MC(MPI_Comm_split(MPI_COMM_WORLD,nodes.node_rank==0?0:MPI_UNDEFINED,world_rank,&nodes.leaders_comm));
}
static void broadcast_from_root(void* buffer,int count,MPI_Datatype type) {
  if (!nodes.hierarchical) { MC(MPI_Bcast(buffer,count,type,0,MPI_COMM_WORLD)); return; }
  if (nodes.node_rank==0) MC(MPI_Bcast(buffer,count,type,0,nodes.leaders_comm));
  MC(MPI_Bcast(buffer,count,type,0,nodes.node_comm));
}
static std::string node_topology_json(const Config& c) {
  std::ostringstream out;
  out<<"{\"grouping\":"<<quote(c.validation_node_size?"validation_only_contiguous_rank_groups":"MPI_COMM_TYPE_SHARED")
     <<",\"simulated\":"<<(c.validation_node_size?"true":"false")<<",\"validation_node_size\":"<<c.validation_node_size
     <<",\"node_group_count\":"<<nodes.group_count<<",\"actual_shared_memory_group_count\":"<<nodes.actual_group_count
     <<",\"leader_world_rank_by_rank\":[";
  for(size_t i=0;i<nodes.leader_by_world_rank.size();++i){if(i)out<<',';out<<nodes.leader_by_world_rank[i];}
  out<<"],\"groups\":[";
  for(size_t i=0;i<nodes.groups.size();++i){if(i)out<<',';out<<'[';for(size_t j=0;j<nodes.groups[i].size();++j){if(j)out<<',';out<<nodes.groups[i][j];}out<<']';}
  out<<"],\"qualification\":\"Logical validation groups are not physical hosts; physical GPU guards always use actual shared-memory communicator\"}";
  return out.str();
}
template<class T> struct DeviceBuffer {
  T* p = nullptr; size_t count = 0;
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t n) { allocate(n); }
  DeviceBuffer(const DeviceBuffer&) = delete; DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() { if (p) cudaFree(p); }
  void allocate(size_t n) { if (count == n) return; if (p) CHECK_CUDA(cudaFree(p)); p = nullptr; count = n; if (n) CHECK_CUDA(cudaMalloc((void**)&p, n*sizeof(T))); }
  void upload(const std::vector<T>& a) { allocate(a.size()); if (!a.empty()) CHECK_CUDA(cudaMemcpy(p, a.data(), a.size()*sizeof(T), cudaMemcpyHostToDevice)); }
  std::vector<T> download() const { std::vector<T> a(count); if (count) CHECK_CUDA(cudaMemcpy(a.data(), p, count*sizeof(T), cudaMemcpyDeviceToHost)); return a; }
  void zero() { if (count) CHECK_CUDA(cudaMemset(p, 0, count*sizeof(T))); }
};
template<class T> static void write_binary(const fs::path& path, const std::vector<T>& values) {
  std::ofstream out(path, std::ios::binary); if (!out) throw std::runtime_error("open output: " + path.string());
  out.write(reinterpret_cast<const char*>(values.data()), (std::streamsize)(values.size()*sizeof(T)));
  if (!out) throw std::runtime_error("write output: " + path.string());
}
static void write_text(const fs::path& path, const std::string& text) { std::ofstream out(path); if (!out) throw std::runtime_error("open output: " + path.string()); out << text << '\n'; if (!out) throw std::runtime_error("write output: " + path.string()); }
static uint64_t file_fingerprint(const Config& c, long long npix) {
  if (c.synthetic) return string_fingerprint("synthetic-v1:1+((17*global_pixel_id+11)%257)/256:npix=" + std::to_string(npix));
  if (fs::file_size(c.input) != (uint64_t)npix*sizeof(float)) throw std::runtime_error("input must be exactly npix little-endian float32 values");
  std::ifstream in(c.input, std::ios::binary); if (!in) throw std::runtime_error("open input: " + c.input);
  std::vector<char> block(1<<20); uint64_t h = 14695981039346656037ULL;
  while (in) { in.read(block.data(), block.size()); h = fnv_update(h, block.data(), (size_t)in.gcount()); }
  if (!in.eof()) throw std::runtime_error("read input fingerprint failed"); return h;
}
static std::vector<float> load_scaled_sky(const Config& c, long long npix, const Ownership& part) {
  std::vector<float> b((size_t)part.pixel_count);
  if (c.synthetic) {
    for (long long i = 0; i < part.pixel_count; ++i) b[i] = 1.0f + (float)(((part.pixel_start+i)*17+11)%257)*(1.0f/256.0f);
  } else {
    std::ifstream in(c.input, std::ios::binary); if (!in) throw std::runtime_error("open input: " + c.input);
    in.seekg(part.pixel_start*(long long)sizeof(float)); in.read(reinterpret_cast<char*>(b.data()), (std::streamsize)(b.size()*sizeof(float)));
    if (in.gcount() != (std::streamsize)(b.size()*sizeof(float))) throw std::runtime_error("short sky shard read");
  }
  const double area = 4.0*M_PI/(double)npix;
  for (float& x : b) { if (!std::isfinite(x)) throw std::runtime_error("non-finite sky value"); x = (float)((double)x*area); }
  return b;
}

struct PixelGPU {
  Ownership part; int ntile;
  DeviceBuffer<float> b,l,m,n,nm1,cx,cy,cz,ca,sa,cacc,cseg;
  DeviceBuffer<uint32_t> wacc,wseg;
  PixelGPU(const Config& c, long long npix, const Ownership& p) : part(p), ntile((int)p.tile_count) {
    b.upload(load_scaled_sky(c,npix,p)); l.allocate(p.pixel_count); m.allocate(p.pixel_count); n.allocate(p.pixel_count); nm1.allocate(p.pixel_count);
    cx.allocate(ntile);cy.allocate(ntile);cz.allocate(ntile);ca.allocate(ntile);sa.allocate(ntile);
    cacc.allocate(p.pixel_count);cseg.allocate(p.pixel_count);wacc.allocate(p.pixel_count);wseg.allocate(p.pixel_count);
    const int blocks=(int)((p.pixel_count+255)/256);
    pix2lmn_nest_kernel<<<blocks,256>>>(c.nside,(unsigned int)p.pixel_start,(int)p.pixel_count,l.p,m.p,n.p);
    build_nm1_kernel<<<blocks,256>>>(n.p,nm1.p,p.pixel_count);
    build_tile_cone_meta_kernel<256><<<ntile,256>>>(l.p,m.p,n.p,p.pixel_count,cx.p,cy.p,cz.p,ca.p,sa.p,ntile);
    CHECK_CUDA(cudaGetLastError());CHECK_CUDA(cudaDeviceSynchronize()); reset();
  }
  void reset() { cacc.zero();wacc.zero(); }
};
struct Geometry { int n=0,nh=0,t0=0,tlen=0; std::array<std::vector<float>,11> a; };
struct GeometryGPU {
  std::array<DeviceBuffer<float>,11> a; DeviceBuffer<float2> viss;
  void upload(const Geometry& g) { for(int j=0;j<11;++j) a[j].upload(g.a[j]); viss.allocate(g.nh); }
  void upload_vis(const std::vector<float2>& v) { viss.upload(v); }
};
static Geometry generate_geometry(OrbitGenCtx& gen, GeometryGPU& dev, int t0, int tlen) {
  orbit_gen_make_segment_async(gen,0,t0,tlen,true);orbit_gen_wait_slot(gen,0);const auto& s=gen.slots[0];
  Geometry g;g.n=s.segN;g.nh=tlen*UNIQUE_BASELINES_PER_T;g.t0=t0;g.tlen=tlen;
  const float* source[11]={s.h_u,s.h_v,s.h_w,s.h_x1,s.h_y1,s.h_z1,nullptr,s.h_x2,s.h_y2,s.h_z2,nullptr};
  for(int j=0;j<11;++j) {g.a[j].resize(g.n);if(source[j]) std::copy(source[j],source[j]+g.n,g.a[j].begin());}
  if(!validate_uvw_halfsym_layout_ptr(g.a[0].data(),g.a[1].data(),g.a[2].data(),g.n)) throw std::runtime_error("production orbit half symmetry failed");
  dev.upload(g);const int blocks=(g.n+255)/256;
  invnorm3_kernel<<<blocks,256>>>(dev.a[3].p,dev.a[4].p,dev.a[5].p,dev.a[6].p,g.n);
  invnorm3_kernel<<<blocks,256>>>(dev.a[7].p,dev.a[8].p,dev.a[9].p,dev.a[10].p,g.n);
  CHECK_CUDA(cudaGetLastError());CHECK_CUDA(cudaDeviceSynchronize());g.a[6]=dev.a[6].download();g.a[10]=dev.a[10].download();return g;
}
static void bcast_geometry(Geometry& g) {
  int meta[4]={g.n,g.nh,g.t0,g.tlen};broadcast_from_root(meta,4,MPI_INT);
  g.n=meta[0];g.nh=meta[1];g.t0=meta[2];g.tlen=meta[3];
  for(auto& a:g.a) {a.resize(g.n);broadcast_from_root(a.data(),g.n,MPI_FLOAT);}
}
static std::vector<float2> partial_visibility(PixelGPU& p, GeometryGPU& d, int nh, float cosphi) {
  DeviceBuffer<float2> output(nh);const int grid=(nh+255)/256;
  viss_partial_all_halfsym_tilecone<256,true><<<grid,256,256*5*sizeof(float)>>>(
    p.b.p,p.l.p,p.m.p,p.n.p,p.nm1.p,p.part.pixel_count,p.cx.p,p.cy.p,p.cz.p,p.ca.p,p.sa.p,p.ntile,
    d.a[0].p,d.a[1].p,d.a[2].p,d.a[3].p,d.a[4].p,d.a[5].p,d.a[6].p,d.a[7].p,d.a[8].p,d.a[9].p,d.a[10].p,nh,cosphi,output.p);
  CHECK_CUDA(cudaGetLastError());CHECK_CUDA(cudaDeviceSynchronize());return output.download();
}
static std::vector<float2> ordered_sum(const std::vector<float2>& parts,int size,int nh) {
  std::vector<float2> out(nh);
  for(int i=0;i<nh;++i) {float re=0,im=0;for(int r=0;r<size;++r){re+=parts[(size_t)r*nh+i].x;im+=parts[(size_t)r*nh+i].y;}out[i]=make_float2(re,im);}return out;
}
static std::vector<float2> ordered_strategy_sum(const std::vector<float2>& parts,int size,int nh) {
  if(!nodes.hierarchical)return ordered_sum(parts,size,nh);
  std::vector<float2> group_sums((size_t)nodes.group_count*nh);
  for(int group=0;group<nodes.group_count;++group)for(int i=0;i<nh;++i){
    float re=0,im=0;
    for(int r:nodes.groups[group]){re+=parts[(size_t)r*nh+i].x;im+=parts[(size_t)r*nh+i].y;}
    group_sums[(size_t)group*nh+i]=make_float2(re,im);
  }
  return ordered_sum(group_sums,nodes.group_count,nh);
}
static std::vector<float2> collect_visibility(const std::vector<float2>& partial,int size,int nh) {
  std::vector<float2> result(nh);
  if(!nodes.hierarchical){
    std::vector<float2> parts;if(world_rank==0)parts.resize((size_t)size*nh);
    MC(MPI_Gather(reinterpret_cast<const float*>(partial.data()),nh*2,MPI_FLOAT,reinterpret_cast<float*>(parts.data()),nh*2,MPI_FLOAT,0,MPI_COMM_WORLD));
    if(world_rank==0)result=ordered_sum(parts,size,nh);
    return result;
  }
  std::vector<float2> local_parts;if(nodes.node_rank==0)local_parts.resize((size_t)nodes.node_size*nh);
  MC(MPI_Gather(reinterpret_cast<const float*>(partial.data()),nh*2,MPI_FLOAT,reinterpret_cast<float*>(local_parts.data()),nh*2,MPI_FLOAT,0,nodes.node_comm));
  if(nodes.node_rank==0){
    auto local_sum=ordered_sum(local_parts,nodes.node_size,nh);
    std::vector<float2> leader_parts;if(world_rank==0)leader_parts.resize((size_t)nodes.group_count*nh);
    MC(MPI_Gather(reinterpret_cast<const float*>(local_sum.data()),nh*2,MPI_FLOAT,reinterpret_cast<float*>(leader_parts.data()),nh*2,MPI_FLOAT,0,nodes.leaders_comm));
    if(world_rank==0)result=ordered_sum(leader_parts,nodes.group_count,nh);
  }
  return result;
}
struct Grid {float fa=0,fb=0,du=0;int res=0,half=0;};
static Grid make_grid(const Geometry& g,const Config& c,float lambda) {
  Grid out;double denom=0;compute_fa_fb_host_matlab_no_intercept(g.a[0].data(),g.a[1].data(),g.a[2].data(),g.n,out.fa,out.fb,denom);
  float lmmax=std::max(std::sqrt(1.0f+out.fa*out.fa),std::sqrt(1.0f+out.fb*out.fb));
  float temp_res=std::ceil(2.0f*100e3f/lambda*2.0f*lmmax);int base_res=(int)(temp_res+2+1-std::fmod(temp_res,2.0f));
  double requested=std::ceil((double)(base_res-1)/(double)c.grid_cell_scale);
  if(!std::isfinite(requested)||requested>INT_MAX-2)throw std::runtime_error("grid interval range");
  int intervals=std::max(2,(int)requested);if(intervals&1)++intervals;out.res=intervals+1;
  if((long long)out.res*out.res>INT_MAX-1LL)throw std::runtime_error("grid keys exceed INT_MAX");
  out.half=intervals/2;out.du=2.0f*100e3f/lambda/(float)intervals;return out;
}
template<class T> static void bcast_vector(std::vector<T>& v,MPI_Datatype type) {
  if(v.size()>(size_t)INT_MAX)fatal("MPI vector count exceeds INT_MAX");int count=(int)v.size();broadcast_from_root(&count,1,MPI_INT);if(count<0)fatal("negative MPI vector count");v.resize(count);broadcast_from_root(v.data(),count,type);
}
static void bcast_group(GroupArtifactsHost& g,Grid& grid) {
  float coeff[3]={grid.fa,grid.fb,grid.du};int ints[3]={grid.res,grid.half,g.nuniq};broadcast_from_root(coeff,3,MPI_FLOAT);broadcast_from_root(ints,3,MPI_INT);
  grid.fa=coeff[0];grid.fb=coeff[1];grid.du=coeff[2];grid.res=ints[0];grid.half=ints[1];g.nuniq=ints[2];
  bcast_vector(g.keys_unique,MPI_INT);bcast_vector(g.counts,MPI_INT);bcast_vector(g.rep_half_idx,MPI_INT);bcast_vector(g.offsets,MPI_INT);bcast_vector(g.idx_sorted,MPI_INT);
}
static GroupArtifactsHost build_group(GeometryGPU& g,const std::vector<float2>& vis,const Grid& grid,int dev,bool rep) {
  g.upload_vis(vis);GroupArtifactsHost group;build_groups_device0_half(dev,nullptr,g.a[0].p,g.a[1].p,g.viss.p,(int)vis.size(),grid.du,grid.res,grid.half,1,rep?2:0,group);return group;
}
static void reconstruct(PixelGPU& p,GeometryGPU& g,const GroupArtifactsHost& group,const Grid& grid,float cosphi,bool rep) {
  if(!group.nuniq)return;
  const int blocks=(int)(p.part.pixel_count/256);
  DeviceBuffer<int> keys,idx,offsets,repidx;keys.upload(group.keys_unique);
  p.cseg.zero();p.wseg.zero();
  if(rep){
    std::vector<float> hu(group.nuniq),hv(group.nuniq);for(int i=0;i<group.nuniq;++i){int key=group.keys_unique[i]-1;int u=key/grid.res;int v=key-u*grid.res;hu[i]=(u-grid.half)*grid.du;hv[i]=(v-grid.half)*grid.du;}
    DeviceBuffer<float> u,v;u.upload(hu);v.upload(hv);repidx.upload(group.rep_half_idx);
    DeviceBuffer<float2> rv(group.nuniq);DeviceBuffer<float4> p1(group.nuniq),p2(group.nuniq);
    gather_rep_half_inputs<<<(group.nuniq+255)/256,256>>>(repidx.p,group.nuniq,g.viss.p,g.a[3].p,g.a[4].p,g.a[5].p,g.a[6].p,g.a[7].p,g.a[8].p,g.a[9].p,g.a[10].p,rv.p,p1.p,p2.p);
    recon_seg_blockage_half_rep<64><<<blocks,256>>>(p.part.pixel_count,p.l.p,p.m.p,p.n.p,keys.p,u.p,v.p,rv.p,p1.p,p2.p,group.nuniq,grid.fa,grid.fb,cosphi,p.cseg.p,p.wseg.p);
    CHECK_CUDA(cudaGetLastError());CHECK_CUDA(cudaDeviceSynchronize());
  }else{
    idx.upload(group.idx_sorted);offsets.upload(group.offsets);
    recon_seg_blockage_half<64><<<blocks,256>>>(p.part.pixel_count,p.l.p,p.m.p,p.n.p,keys.p,offsets.p,idx.p,g.viss.p,g.a[3].p,g.a[4].p,g.a[5].p,g.a[6].p,g.a[7].p,g.a[8].p,g.a[9].p,g.a[10].p,group.nuniq,grid.res,grid.half,grid.du,grid.fa,grid.fb,cosphi,0,p.cseg.p,p.wseg.p);
    CHECK_CUDA(cudaGetLastError());CHECK_CUDA(cudaDeviceSynchronize());
  }
  add_inplace<<<blocks,256>>>(p.cacc.p,p.cseg.p,p.part.pixel_count);add_inplace_u32<<<blocks,256>>>(p.wacc.p,p.wseg.p,p.part.pixel_count);
  CHECK_CUDA(cudaGetLastError());CHECK_CUDA(cudaDeviceSynchronize());
}
struct Image {std::vector<float> numerator,map;std::vector<uint32_t> weight;};
static Image finish_image(PixelGPU& p) {
  Image x;x.numerator=p.cacc.download();x.weight=p.wacc.download();
  normalize_by_weight<<<(int)(p.part.pixel_count/256),256>>>(p.cacc.p,p.wacc.p,p.part.pixel_count);CHECK_CUDA(cudaGetLastError());CHECK_CUDA(cudaDeviceSynchronize());x.map=p.cacc.download();return x;
}
static void write_image(const fs::path& dir,const Image& x) {fs::create_directories(dir);write_binary(dir/"numerator_f32.bin",x.numerator);write_binary(dir/"weight_u32.bin",x.weight);write_binary(dir/"map_f32.bin",x.map);}
static std::string image_checksum_json(const Image& x) {
  const uint64_t initial=14695981039346656037ULL;
  std::ostringstream out;out<<"{\"numerator_f32\":"<<quote(hex64(fnv_update(initial,x.numerator.data(),x.numerator.size()*sizeof(float))))
    <<",\"weight_u32\":"<<quote(hex64(fnv_update(initial,x.weight.data(),x.weight.size()*sizeof(uint32_t))))
    <<",\"map_f32\":"<<quote(hex64(fnv_update(initial,x.map.data(),x.map.size()*sizeof(float))))<<'}';
  return out.str();
}
static void write_image_audit(const fs::path& path,const Image& x,bool binary_written,const std::string& checksums) {
  const size_t count=std::min<size_t>(4096,x.map.size());
  auto index=[&](size_t i){return count>1?i*(x.map.size()-1)/(count-1):size_t(0);};
  std::ostringstream out;out<<std::setprecision(std::numeric_limits<float>::max_digits10)
    <<"{\"schema\":\"sts_image_audit_v1\",\"npix\":"<<x.map.size()<<",\"image_binary_written\":"<<(binary_written?"true":"false")
    <<",\"checksum_algorithm\":\"fnv1a64_raw_array_bytes\",\"checksum_qualification\":\"Noncryptographic consistency guard over complete native little-endian arrays; not scientific accuracy evidence or a replacement for artifact SHA256\",\"array_checksums\":"<<checksums
    <<",\"samples\":{\"count\":"<<count<<",\"selection\":\"floor(i*(npix-1)/(count-1))\",\"indices\":[";
  for(size_t i=0;i<count;++i){if(i)out<<',';out<<index(i);}out<<']';
  auto emit=[&](const auto& values){out<<'[';for(size_t i=0;i<count;++i){if(i)out<<',';out<<values[index(i)];}out<<']';};
  out<<",\"numerator_f32\":";emit(x.numerator);out<<",\"weight_u32\":";emit(x.weight);out<<",\"map_f32\":";emit(x.map);out<<"}}";
  write_text(path,out.str());
}
static std::string host_rss_json() {
  std::ostringstream out;out<<"{\"method\":\"getrusage(RUSAGE_SELF).ru_maxrss\",\"qualification\":\"Process lifetime resident-set high-water; shared pages may be counted by multiple ranks; not node-total RAM\",";
#if defined(__linux__)
  struct rusage usage{};
  if(getrusage(RUSAGE_SELF,&usage)==0){out<<"\"available\":true,\"peak_bytes\":"<<(uint64_t(usage.ru_maxrss)*1024ULL)<<",\"native_unit\":\"KiB\"}";return out.str();}
#endif
  out<<"\"available\":false,\"peak_bytes\":null}";return out.str();
}
static uint64_t group_metadata_payload_bytes(const GroupArtifactsHost& g) {
  return 3ULL*sizeof(float)+(3ULL+5ULL)*sizeof(int)+
    (uint64_t(g.keys_unique.size())+g.counts.size()+g.rep_half_idx.size()+g.offsets.size()+g.idx_sorted.size())*sizeof(int);
}
template<class T>static bool bitwise_equal(const std::vector<T>& a,const std::vector<T>& b){return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);}
template<class T>static std::vector<T> gather_pixels(const std::vector<T>& local,long long npix,int size,MPI_Datatype type,int columns=1) {
  std::vector<int> counts(size),offsets(size);for(int r=0;r<size;++r){auto p=ownership(npix,r,size);long long count=p.pixel_count*columns,offset=p.pixel_start*columns;if(count>INT_MAX||offset>INT_MAX)fatal("MPI output count/displacement limit");counts[r]=(int)count;offsets[r]=(int)offset;}
  std::vector<T> out;if(world_rank==0)out.resize((size_t)npix*columns);
  MC(MPI_Gatherv(local.data(),(int)local.size(),type,out.data(),counts.data(),offsets.data(),type,0,MPI_COMM_WORLD));return out;
}
static void dump_pixels(PixelGPU& p,long long npix,int size,const fs::path& daydir) {
  auto b=p.b.download(),l=p.l.download(),m=p.m.download(),n=p.n.download(),nm1=p.nm1.download();std::vector<float> rows((size_t)p.part.pixel_count*5);
  for(size_t i=0;i<b.size();++i){rows[5*i]=b[i];rows[5*i+1]=l[i];rows[5*i+2]=m[i];rows[5*i+3]=n[i];rows[5*i+4]=nm1[i];}
  auto all=gather_pixels(rows,npix,size,MPI_FLOAT,5);if(world_rank==0)write_binary(daydir/"pixels_f32.bin",all);
}
static void dump_segment(const fs::path& dir,const Config& c,int day,int segment,const Geometry& g,const Grid& grid,const GroupArtifactsHost& group,const std::vector<float2>& vis,float cosphi) {
  fs::create_directories(dir);std::vector<float> rows((size_t)g.n*11);for(int i=0;i<g.n;++i)for(int j=0;j<11;++j)rows[11*(size_t)i+j]=g.a[j][i];
  write_binary(dir/"geometry_f32.bin",rows);write_binary(dir/"visibility_half_f32.bin",vis);write_binary(dir/"keys_i32.bin",group.keys_unique);write_binary(dir/"counts_i32.bin",group.counts);
  write_binary(dir/"rep_half_idx_i32.bin",group.rep_half_idx);write_binary(dir/"offsets_i32.bin",group.offsets);write_binary(dir/"idx_sorted_i32.bin",group.idx_sorted);
  std::ostringstream j;j<<std::setprecision(17)<<"{\"schema\":\"sts_cluster_segment_dump_v1\",\"day\":"<<day<<",\"segment_index\":"<<segment<<",\"t0\":"<<g.t0<<",\"tlen\":"<<g.tlen<<",\"segN\":"<<g.n<<",\"segN_half\":"<<g.nh<<",\"nuniq\":"<<group.nuniq<<",\"RES\":"<<grid.res<<",\"half\":"<<grid.half<<",\"du\":"<<grid.du<<",\"fa\":"<<grid.fa<<",\"fb\":"<<grid.fb<<",\"cosphi\":"<<cosphi<<",\"recon_mode\":"<<quote(c.recon_mode)<<",\"geometry_columns\":[\"u\",\"v\",\"w\",\"x1\",\"y1\",\"z1\",\"invn1\",\"x2\",\"y2\",\"z2\",\"invn2\"],\"pixel_columns\":[\"bp32\",\"l\",\"m\",\"n\",\"nm1\"],\"dtype\":\"little_endian_float32\",\"reconstruction_coordinates\":\"l+fa*n,m+fb*n_archived_production_convention\",\"half_layout\":\"time_major_28_positive_then28_negative\"}";write_text(dir/"metadata.json",j.str());
}

static std::vector<std::string> gather_strings(const std::string& local,int size) {
  int length=(int)local.size();std::vector<int> lengths(size),offsets(size);MC(MPI_Gather(&length,1,MPI_INT,lengths.data(),1,MPI_INT,0,MPI_COMM_WORLD));int total=0;
  if(world_rank==0)for(int r=0;r<size;++r){offsets[r]=total;total+=lengths[r];}
  std::vector<char> bytes(world_rank==0?total:0);MC(MPI_Gatherv(local.data(),length,MPI_CHAR,bytes.data(),lengths.data(),offsets.data(),MPI_CHAR,0,MPI_COMM_WORLD));
  std::vector<std::string> out;if(world_rank==0)for(int r=0;r<size;++r)out.emplace_back(bytes.data()+offsets[r],lengths[r]);return out;
}
struct SerialStatus {bool visibility=true,numerator=true,weight=true,map=true;bool passed()const{return visibility&&numerator&&weight&&map;}};
struct RunMeasurements {
  double compute_loop_s=0,geometry_root_s=0,geometry_broadcast_and_upload_s=0,visibility_local_gpu_d2h_s=0;
  double visibility_gather_sum_phase_s=0,group_root_s=0,metadata_broadcast_and_visibility_upload_s=0,reconstruction_local_s=0;
  double image_finish_download_s=0,image_gather_s=0,image_binary_write_s=0,image_finite_validation_s=0,image_audit_hash_sample_s=0;
  uint64_t segment_count=0,geometry_logical_payload_bytes=0,visibility_broadcast_logical_payload_bytes=0,group_metadata_logical_payload_bytes=0;
  std::string json(double elapsed_s)const{
    std::ostringstream out;out<<std::setprecision(17)
      <<"{\"compute_loop_s\":"<<compute_loop_s<<",\"compute_loop_definition\":\"Sum of this rank's segment_wall_s; includes stage-boundary instrumentation and optional tiny validation, excludes daily finish/download/gather/image writing/finite validation/hash samples; not pure GPU kernel time\",\"elapsed_s\":"<<elapsed_s
      <<",\"segment_count\":"<<segment_count<<",\"geometry_root_s\":"<<geometry_root_s<<",\"geometry_broadcast_and_upload_s\":"<<geometry_broadcast_and_upload_s
      <<",\"visibility_local_gpu_d2h_s\":"<<visibility_local_gpu_d2h_s<<",\"visibility_gather_sum_phase_s\":"<<visibility_gather_sum_phase_s
      <<",\"group_root_s\":"<<group_root_s<<",\"metadata_broadcast_and_visibility_upload_s\":"<<metadata_broadcast_and_visibility_upload_s
      <<",\"reconstruction_local_s\":"<<reconstruction_local_s<<",\"image_finish_download_s\":"<<image_finish_download_s
      <<",\"image_gather_s\":"<<image_gather_s<<",\"image_binary_write_s\":"<<image_binary_write_s
      <<",\"image_finite_validation_s\":"<<image_finite_validation_s<<",\"image_audit_hash_sample_s\":"<<image_audit_hash_sample_s
      <<",\"geometry_logical_payload_bytes\":"<<geometry_logical_payload_bytes<<",\"visibility_broadcast_logical_payload_bytes\":"<<visibility_broadcast_logical_payload_bytes
      <<",\"group_metadata_logical_payload_bytes\":"<<group_metadata_logical_payload_bytes
      <<",\"payload_qualification\":\"Logical broadcast argument bytes, counted once per segment; not per-recipient traffic, MPI internal traffic or measured network bytes. Geometry includes 4 metadata ints and 11 float vectors; group metadata includes scalars, 5 vector lengths and actual vector entries\",\"stage_timing_qualification\":\"Visibility includes output allocation/kernel synchronization/D2H; reconstruction includes host preparation and device allocation/copies/kernels; no pure CUDA event kernel timing\"}";
    return out.str();
  }
};

static int run(int argc,char** argv) {
  Config c=parse_config(argc,argv);if(c.help){if(world_rank==0)usage();return 0;}
  int size=0;MC(MPI_Comm_size(MPI_COMM_WORLD,&size));long long npix=12LL*c.nside*c.nside;
  if(size>npix/TILE_PIX)throw std::runtime_error("more ranks than whole pixel tiles");
  auto config=config_json(c);uint64_t confhash=string_fingerprint(config);std::vector<uint64_t> hashes(size);MC(MPI_Allgather(&confhash,1,MPI_UINT64_T,hashes.data(),1,MPI_UINT64_T,MPI_COMM_WORLD));if(!std::all_of(hashes.begin(),hashes.end(),[&](uint64_t x){return x==confhash;}))throw std::runtime_error("ranks received different configs");
  uint64_t inputhash=file_fingerprint(c,npix);MC(MPI_Allgather(&inputhash,1,MPI_UINT64_T,hashes.data(),1,MPI_UINT64_T,MPI_COMM_WORLD));if(!std::all_of(hashes.begin(),hashes.end(),[&](uint64_t x){return x==inputhash;}))throw std::runtime_error("input file fingerprint differs across ranks");
  MPI_Comm local_comm;MC(MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,world_rank,MPI_INFO_NULL,&local_comm));int local_rank=0,local_size=0;MC(MPI_Comm_rank(local_comm,&local_rank));MC(MPI_Comm_size(local_comm,&local_size));
  int device_count=0;CHECK_CUDA(cudaGetDeviceCount(&device_count));if(device_count<1)throw std::runtime_error("no visible CUDA GPU");
  int device=device_count==1?0:local_rank%device_count;CHECK_CUDA(cudaSetDevice(device));cudaDeviceProp prop{};CHECK_CUDA(cudaGetDeviceProperties(&prop,device));
  std::array<unsigned char,16> uuid{};std::memcpy(uuid.data(),prop.uuid.bytes,16);std::vector<unsigned char> local_uuids((size_t)local_size*16);MC(MPI_Allgather(uuid.data(),16,MPI_UNSIGNED_CHAR,local_uuids.data(),16,MPI_UNSIGNED_CHAR,local_comm));
  bool duplicate_gpu=false;for(int a=0;a<local_size;++a)for(int b=a+1;b<local_size;++b)if(std::memcmp(local_uuids.data()+a*16,local_uuids.data()+b*16,16)==0)duplicate_gpu=true;
  if(duplicate_gpu&&!c.allow_shared_gpu)throw std::runtime_error("multiple local ranks share one physical GPU; explicit --allow-shared-gpu=1 is required for correctness smoke only");
  initialize_node_transport(c,local_comm,local_rank,size);
  int driver=0,runtime=0;CHECK_CUDA(cudaDriverGetVersion(&driver));CHECK_CUDA(cudaRuntimeGetVersion(&runtime));
  char hostname[MPI_MAX_PROCESSOR_NAME];int hlen=0;MC(MPI_Get_processor_name(hostname,&hlen));std::string host(hostname,hlen);
  auto hosts=gather_strings(host,size);auto part=ownership(npix,world_rank,size);
  size_t free_bytes=0,total_bytes=0;CHECK_CUDA(cudaMemGetInfo(&free_bytes,&total_bytes));
  DeviceMemoryObservations memory_observations(free_bytes,total_bytes);
  uint64_t pixel_arrays=(uint64_t)part.pixel_count*36ULL,minimum_budget=pixel_arrays+(512ULL<<20);
  if(minimum_budget>free_bytes)throw std::runtime_error("pixel arrays plus512MiB reserve exceed currently free device memory; reduce nside or use more real GPU ranks");
  fs::path out=c.output_dir;if(fs::exists(out/"metrics.json"))throw std::runtime_error("output already has metrics.json; use a new run directory");
  for(int d=0;d<c.day_count;++d)if(fs::exists(out/("day"+padded(c.day_start+d))))throw std::runtime_error("day output exists; do not overwrite prior results");
  fs::create_directories(out);MC(MPI_Barrier(MPI_COMM_WORLD));
  std::ostringstream rankj;rankj<<"{\"rank\":"<<world_rank<<",\"hostname\":"<<quote(host)<<",\"local_rank\":"<<local_rank<<",\"local_size\":"<<local_size<<",\"logical_gpu\":"<<device<<",\"gpu_name\":"<<quote(prop.name)<<",\"compute_capability\":"<<quote(std::to_string(prop.major)+"."+std::to_string(prop.minor))<<",\"gpu_uuid\":\"";for(auto b:uuid)rankj<<std::hex<<std::setw(2)<<std::setfill('0')<<(unsigned)b;rankj<<std::dec<<"\",\"cuda_driver\":"<<driver<<",\"cuda_runtime\":"<<runtime<<",\"initial_free_device_bytes\":"<<free_bytes<<",\"device_total_bytes\":"<<total_bytes<<",\"pixel_array_bytes\":"<<pixel_arrays<<",\"pixel_start\":"<<part.pixel_start<<",\"pixel_count\":"<<part.pixel_count<<",\"same_gpu_shared_locally\":"<<(duplicate_gpu?"true":"false")<<"}";
  write_text(out/("rank"+padded(world_rank)+".json"),rankj.str());auto rankinfo=gather_strings(rankj.str(),size);
  if(world_rank==0){std::ostringstream csv;csv<<"rank,pixel_start,pixel_count,tile_start,tile_count\n";for(int r=0;r<size;++r){auto p=ownership(npix,r,size);csv<<r<<','<<p.pixel_start<<','<<p.pixel_count<<','<<p.tile_start<<','<<p.tile_count<<'\n';}write_text(out/"ownership.csv",csv.str());}
  PixelGPU pixels(c,npix,part);GeometryGPU gd;
  memory_observations.capture("after_pixel_allocation");
  std::vector<std::unique_ptr<PixelGPU>> serial_pixels;if(world_rank==0&&c.serial_check)for(int r=0;r<size;++r)serial_pixels.emplace_back(new PixelGPU(c,npix,ownership(npix,r,size)));
  if(world_rank==0&&c.serial_check)memory_observations.capture("after_validation_replay_allocation");
  float lambda=c.btag=="1M"?300.0f:c.btag=="10M"?30.0f:10.0f;
  float R=1737.1e3f,h=300e3f,theta=asinf(R/(R+h)),phi=(float)M_PI-theta,cosphi=cosf(phi);
  int orbit_res=(int)std::ceil((double)(2.0*M_PI)*(double)(100e3/(double)lambda));int procession=round_away_from_zero_host(24.0f/ORBIT_HOURS);int full_t=procession*(orbit_res/3);
  if(c.time_start>=full_t)throw std::runtime_error("time_start outside production full-day range");int selected_t=c.time_count?std::min(c.time_count,full_t-c.time_start):full_t-c.time_start;
  int effective_segs=std::min(c.segs,selected_t),selected_segments=c.segments_limit?std::min(effective_segs,c.segments_limit):effective_segs;
  std::vector<int> starts(effective_segs),lengths(effective_segs);int current=c.time_start,max_tlen=0;
  for(int s=0;s<effective_segs;++s){int len=selected_t/effective_segs+(s<selected_t%effective_segs);starts[s]=current;current+=len;if(c.tlen_limit)len=std::min(len,c.tlen_limit);lengths[s]=len;if(s<selected_segments)max_tlen=std::max(max_tlen,len);}
  if((long long)max_tlen*SIGNED_BASELINES_PER_T>INT_MAX || (long long)max_tlen*UNIQUE_BASELINES_PER_T*2*size>INT_MAX)throw std::runtime_error("segment or visibility gather exceeds supported MPI/int range");
  std::ostringstream profile;profile<<"day,segment,t0,tlen,segN_half,nuniq,geometry_root_s,geometry_broadcast_s,visibility_local_gpu_d2h_s,visibility_gather_sum_phase_s,group_root_s,metadata_broadcast_s,reconstruction_local_s,segment_wall_s,geometry_logical_payload_bytes,visibility_broadcast_logical_payload_bytes,group_metadata_logical_payload_bytes,group_keys_count,group_counts_count,group_rep_idx_count,group_offsets_count,group_idx_sorted_count\n";profile<<std::setprecision(17);
  RunMeasurements measured;
  std::vector<std::string> day_json;SerialStatus serial;MC(MPI_Barrier(MPI_COMM_WORLD));double total_start=MPI_Wtime();
  for(int di=0;di<c.day_count;++di){
    int day=c.day_start+di;fs::path daydir=out/("day"+padded(day));fs::create_directories(daydir);pixels.reset();for(auto& p:serial_pixels)p->reset();
    if(c.dump_validation)dump_pixels(pixels,npix,size,daydir);
    OrbitGenCtx gen;if(world_rank==0)orbit_gen_init(gen,device,lambda,day,max_tlen,c.orbit_seed);
    memory_observations.capture("after_orbit_initialization",day);
    long long total_half=0,total_groups=0;double day_start=MPI_Wtime(),day_compute_start=measured.compute_loop_s;
    for(int s=0;s<selected_segments;++s){
      MC(MPI_Barrier(MPI_COMM_WORLD));double seg_start=MPI_Wtime(),start=MPI_Wtime();Geometry g;
      if(world_rank==0)g=generate_geometry(gen,gd,starts[s],lengths[s]);double geometry_s=MPI_Wtime()-start;
      start=MPI_Wtime();bcast_geometry(g);gd.upload(g);double geometry_bcast_s=MPI_Wtime()-start;
      memory_observations.capture("after_geometry_upload",day,s);
      start=MPI_Wtime();auto partial=partial_visibility(pixels,gd,g.nh,cosphi);double visibility_s=MPI_Wtime()-start;
      start=MPI_Wtime();auto vis=collect_visibility(partial,size,g.nh);std::vector<float2> reference_vis;
      if(world_rank==0)phase_correct_viss_halfsym_host(vis.data(),g.a[2].data(),g.nh);
      double reduction_s=MPI_Wtime()-start;
      if(world_rank==0&&c.serial_check){std::vector<float2> ref_parts((size_t)size*g.nh);for(int r=0;r<size;++r){auto z=partial_visibility(*serial_pixels[r],gd,g.nh,cosphi);std::copy(z.begin(),z.end(),ref_parts.begin()+(size_t)r*g.nh);}reference_vis=ordered_strategy_sum(ref_parts,size,g.nh);phase_correct_viss_halfsym_host(reference_vis.data(),g.a[2].data(),g.nh);serial.visibility=serial.visibility&&bitwise_equal(vis,reference_vis);}
      start=MPI_Wtime();broadcast_from_root(reinterpret_cast<float*>(vis.data()),g.nh*2,MPI_FLOAT);gd.upload_vis(vis);double metadata_s=MPI_Wtime()-start;
      memory_observations.capture("after_visibility_upload",day,s);
      Grid grid;GroupArtifactsHost group;start=MPI_Wtime();if(world_rank==0){grid=make_grid(g,c,lambda);group=build_group(gd,vis,grid,device,c.recon_mode=="rep");}double group_s=MPI_Wtime()-start;
      start=MPI_Wtime();bcast_group(group,grid);metadata_s+=MPI_Wtime()-start;
      uint64_t geometry_payload=4ULL*sizeof(int)+11ULL*(uint64_t)g.n*sizeof(float);
      uint64_t visibility_payload=(uint64_t)g.nh*sizeof(float2),group_payload=group_metadata_payload_bytes(group);
      if(c.dump_validation&&world_rank==0)dump_segment(daydir/("segment"+padded(s)),c,day,s,g,grid,group,vis,cosphi);
      memory_observations.capture("before_reconstruction",day,s);
      start=MPI_Wtime();reconstruct(pixels,gd,group,grid,cosphi,c.recon_mode=="rep");double recon_s=MPI_Wtime()-start;
      memory_observations.capture("after_reconstruction",day,s);
      if(world_rank==0&&c.serial_check){auto ref_group=build_group(gd,reference_vis,grid,device,c.recon_mode=="rep");for(auto& p:serial_pixels)reconstruct(*p,gd,ref_group,grid,cosphi,c.recon_mode=="rep");gd.upload_vis(vis);}
      total_half+=g.nh;total_groups+=group.nuniq;double wall=MPI_Wtime()-seg_start;
      measured.compute_loop_s+=wall;++measured.segment_count;measured.geometry_root_s+=geometry_s;
      measured.geometry_broadcast_and_upload_s+=geometry_bcast_s;measured.visibility_local_gpu_d2h_s+=visibility_s;
      measured.visibility_gather_sum_phase_s+=reduction_s;measured.group_root_s+=group_s;
      measured.metadata_broadcast_and_visibility_upload_s+=metadata_s;measured.reconstruction_local_s+=recon_s;
      measured.geometry_logical_payload_bytes+=geometry_payload;measured.visibility_broadcast_logical_payload_bytes+=visibility_payload;
      measured.group_metadata_logical_payload_bytes+=group_payload;
      profile<<day<<','<<s<<','<<g.t0<<','<<g.tlen<<','<<g.nh<<','<<group.nuniq<<','<<geometry_s<<','<<geometry_bcast_s<<','<<visibility_s<<','<<reduction_s<<','<<group_s<<','<<metadata_s<<','<<recon_s<<','<<wall
        <<','<<geometry_payload<<','<<visibility_payload<<','<<group_payload<<','<<group.keys_unique.size()<<','<<group.counts.size()<<','<<group.rep_half_idx.size()<<','<<group.offsets.size()<<','<<group.idx_sorted.size()<<'\n';
    }
    if(world_rank==0)orbit_gen_destroy(gen);
    double image_start=MPI_Wtime();auto local=finish_image(pixels);measured.image_finish_download_s+=MPI_Wtime()-image_start;
    if(c.write_images){image_start=MPI_Wtime();write_image(daydir/("rank"+padded(world_rank)),local);measured.image_binary_write_s+=MPI_Wtime()-image_start;}
    image_start=MPI_Wtime();
    Image global;global.numerator=gather_pixels(local.numerator,npix,size,MPI_FLOAT);global.weight=gather_pixels(local.weight,npix,size,MPI_UINT32_T);global.map=gather_pixels(local.map,npix,size,MPI_FLOAT);
    measured.image_gather_s+=MPI_Wtime()-image_start;
    if(world_rank==0){
      if(c.write_images){image_start=MPI_Wtime();write_image(daydir,global);measured.image_binary_write_s+=MPI_Wtime()-image_start;}
      if(c.serial_check){Image reference;for(auto& p:serial_pixels){auto x=finish_image(*p);reference.numerator.insert(reference.numerator.end(),x.numerator.begin(),x.numerator.end());reference.weight.insert(reference.weight.end(),x.weight.begin(),x.weight.end());reference.map.insert(reference.map.end(),x.map.begin(),x.map.end());}serial.numerator=serial.numerator&&bitwise_equal(global.numerator,reference.numerator);serial.weight=serial.weight&&bitwise_equal(global.weight,reference.weight);serial.map=serial.map&&bitwise_equal(global.map,reference.map);write_image(daydir/"serial_reference",reference);}
      image_start=MPI_Wtime();
      size_t finite=0,weighted=0;for(size_t i=0;i<global.map.size();++i){if(std::isfinite(global.map[i])&&std::isfinite(global.numerator[i]))++finite;if(global.weight[i])++weighted;}if(finite!=global.map.size())throw std::runtime_error("nonfinite output");
      measured.image_finite_validation_s+=MPI_Wtime()-image_start;image_start=MPI_Wtime();
      std::string checksums=image_checksum_json(global);write_image_audit(daydir/"image_audit.json",global,c.write_images,checksums);
      measured.image_audit_hash_sample_s+=MPI_Wtime()-image_start;
      std::ostringstream d;d<<std::setprecision(17)<<"{\"day\":"<<day<<",\"dir\":"<<quote("day"+padded(day))<<",\"segments_evaluated\":"<<selected_segments<<",\"half_baselines_evaluated\":"<<total_half<<",\"groups_total\":"<<total_groups<<",\"finite_pixels\":"<<finite<<",\"pixels_with_weight\":"<<weighted<<",\"elapsed_s\":"<<MPI_Wtime()-day_start
        <<",\"compute_loop_s\":"<<measured.compute_loop_s-day_compute_start<<",\"image_binary_written\":"<<(c.write_images?"true":"false")
        <<",\"global_array_fnv1a64\":"<<checksums<<",\"image_audit\":"<<quote("day"+padded(day)+"/image_audit.json")<<"}";day_json.push_back(d.str());
    }
    memory_observations.capture("after_daily_output",day);
  }
  MC(MPI_Barrier(MPI_COMM_WORLD));double elapsed=MPI_Wtime()-total_start;write_text(out/("profile_rank"+padded(world_rank)+".csv"),profile.str());
  memory_observations.capture("after_all_output");
  std::string memory_json="{\"rank_info\":"+rankj.str()+",\"observations\":"+memory_observations.json()+",\"host_rss\":"+host_rss_json()+"}";
  write_text(out/("memory_rank"+padded(world_rank)+".json"),memory_json);
  auto memory_by_rank=gather_strings(memory_json,size);
  if(world_rank==0){std::ostringstream summary;summary<<"{\"schema\":\"sts_cluster_stage_boundary_memory_v1\",\"absolute_peak_measured\":false,\"ranks\":[";for(size_t i=0;i<memory_by_rank.size();++i){if(i)summary<<',';summary<<memory_by_rank[i];}summary<<"]}";write_text(out/"memory_summary.json",summary.str());}
  std::string timing_json="{\"rank_info\":"+rankj.str()+",\"measurements\":"+measured.json(elapsed)+"}";
  write_text(out/("timing_rank"+padded(world_rank)+".json"),timing_json);auto timing_by_rank=gather_strings(timing_json,size);
  double max_rank_compute_loop_s=0;MC(MPI_Reduce(&measured.compute_loop_s,&max_rank_compute_loop_s,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD));
  if(world_rank==0){std::ostringstream summary;summary<<std::setprecision(17)<<"{\"schema\":\"sts_cluster_stage_timing_v1\",\"max_rank_compute_loop_s\":"<<max_rank_compute_loop_s
      <<",\"qualification\":\"Maximum of per-rank segment-wall sums; not a sum of per-segment maxima or pure GPU time. Daily image handling and audit hashing are separate\",\"ranks\":[";
    for(size_t i=0;i<timing_by_rank.size();++i){if(i)summary<<',';summary<<timing_by_rank[i];}summary<<"]}";write_text(out/"timing_summary.json",summary.str());}
  int check_pass=(world_rank!=0||!c.serial_check||serial.passed())?1:0;MC(MPI_Bcast(&check_pass,1,MPI_INT,0,MPI_COMM_WORLD));
  if(world_rank==0){
    std::set<std::string> unique_hosts(hosts.begin(),hosts.end());char mpi_version[MPI_MAX_LIBRARY_VERSION_STRING];int mlen=0;MC(MPI_Get_library_version(mpi_version,&mlen));
    std::ostringstream metrics;metrics<<std::setprecision(17)<<"{\"schema\":\"sts_cluster_mpi_v2\",\"status\":"<<quote(check_pass?"completed":"failed_serial_check")<<",\"config\":"<<config<<",\"ranks\":"<<size<<",\"host_count\":"<<unique_hosts.size()<<",\"actual_host_count\":"<<unique_hosts.size()<<",\"host_count_qualification\":\"unique MPI_Get_processor_name strings; not physical-machine attestation\",\"cross_node_execution\":"<<((!c.validation_node_size&&unique_hosts.size()>1)?"true":"false")<<",\"strategy\":"<<quote(c.strategy)<<",\"node_topology\":"<<node_topology_json(c)<<",\"execution_scope\":"<<quote(c.validation_node_size?(unique_hosts.size()==1?"single_host_simulated_node_groups":"multi_host_simulated_node_groups"):(unique_hosts.size()==1?"single_host_multi_process":"multi_host"))<<",\"mpi_library\":"<<quote(std::string(mpi_version,mlen))<<",\"compiled_cuda_version\":"<<CUDART_VERSION<<",\"input_fingerprint_fnv1a64\":"<<quote(hex64(inputhash))<<",\"input_fingerprint_qualification\":\"consistency guard, not cryptographic SHA256; deployment manifest must bind SHA256\",\"npix\":"<<npix<<",\"tile_pix\":256,\"elapsed_s\":"<<elapsed<<",\"compute_loop_s\":"<<measured.compute_loop_s<<",\"max_rank_compute_loop_s\":"<<max_rank_compute_loop_s<<",\"compute_loop_definition\":\"Root sum of segment_wall_s; excludes daily image handling and hashing; see timing_summary.json for all ranks and stage definitions\",\"timing_summary\":\"timing_summary.json\",\"continuous_resource_monitoring\":false,\"performance_claim\":false,\"normalization\":\"daily_sum_of_segment_group_numerators_divided_by_daily_valid_group_count\",\"reconstruction_coordinates\":\"archived_l+fa*n_m+fb*n_not_nm1\",\"visibility_reduction\":"<<quote(c.strategy=="flat"?"CPU_buffer_MPI_Gather_rank_order_FP32_sum_then_original_host_phase":"CPU_buffer_ordered_local_group_sum_then_ordered_leader_sum_then_original_host_phase")<<",\"float_reduction_order_qualification\":\"hierarchical regrouping is a distinct rounding-order candidate; no arithmetic kernel changes\",\"actual_total_peak_gpu_memory_measured\":false,\"serial_check\":{\"requested\":"<<(c.serial_check?"true":"false")<<",\"passed\":"<<(c.serial_check?(serial.passed()?"true":"false"):"null")<<",\"visibility_bitwise\":"<<(c.serial_check?(serial.visibility?"true":"false"):"null")<<",\"numerator_bitwise\":"<<(c.serial_check?(serial.numerator?"true":"false"):"null")<<",\"weight_bitwise\":"<<(c.serial_check?(serial.weight?"true":"false"):"null")<<",\"map_bitwise\":"<<(c.serial_check?(serial.map?"true":"false"):"null")<<"},\"rank_info\":[";
    for(size_t i=0;i<rankinfo.size();++i){if(i)metrics<<',';metrics<<rankinfo[i];}metrics<<"],\"days\":[";for(size_t i=0;i<day_json.size();++i){if(i)metrics<<',';metrics<<day_json[i];}metrics<<"],\"limits\":[\"No genuine cross-node validation until rented hosts are supplied\",\"One GPU/rank, synchronous host-buffer transport; not NCCL or GPU-aware MPI\",\"Root gathers full output and builds groups; root memory and metadata duplication are not eliminated\",\"Pixel partition and hierarchical regrouping change FP32 forward sum ordering\",\"Daily outputs reset accumulators; multiple days do not produce450day cumulative map\",\"Original STS operator and numerical conventions preserved, not a corrected common3D estimator\"]}";
    write_text(out/"metrics.json.tmp",metrics.str());fs::rename(out/"metrics.json.tmp",out/"metrics.json");std::cout<<"status="<<(check_pass?"completed":"failed_serial_check")<<" ranks="<<size<<" hosts="<<unique_hosts.size()<<" npix="<<npix<<" serial_check="<<(c.serial_check?(check_pass?"PASS":"FAIL"):"not_requested")<<std::endl;
  }
  if(nodes.leaders_comm!=MPI_COMM_NULL)MC(MPI_Comm_free(&nodes.leaders_comm));
  MC(MPI_Comm_free(&nodes.node_comm));MC(MPI_Comm_free(&local_comm));return check_pass?0:2;
}
int main(int argc,char** argv) {
  int provided=0;int rc=MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);if(rc!=MPI_SUCCESS){std::cerr<<"MPI_Init_thread failed\n";return 1;}MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);MPI_Comm_set_errhandler(MPI_COMM_WORLD,MPI_ERRORS_RETURN);
  int result=0;try{if(provided<MPI_THREAD_FUNNELED)throw std::runtime_error("MPI thread support below FUNNELED");result=run(argc,argv);}catch(const std::exception& e){fatal(e.what());}catch(...){fatal("unknown exception");}
  MC(MPI_Finalize());return result;
}
