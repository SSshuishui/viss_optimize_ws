// CPU-only MPI transport diagnostic. No CUDA, external input, or monitoring.
// Build with the SAME MPI wrapper/runtime as the application:
//   mpicxx -O2 -std=c++17 tests/mpi_transport_probe.cpp -o build/mpi_transport_probe
// Run under an external timeout; MPI failures are not safely recoverable:
//   timeout 120s mpiexec -n 2 ./build/mpi_transport_probe --iterations=4 > probe.json
// A successful run emits one JSON document on world rank0. Every rank reports
// MPI errors to stderr, then MPI_Abort prevents peers waiting indefinitely.
// --output=PATH also checkpoints root JSON after each completed case.
#include <mpi.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static int world_rank = 0;
static std::string phase = "startup";

static std::string quote(const std::string& s) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') out << '\\' << char(c);
        else if (c == '\n') out << "\\n";
        else if (c == '\r') out << "\\r";
        else if (c == '\t') out << "\\t";
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else out << char(c);
    }
    out << '"';
    return out.str();
}

[[noreturn]] static void fail(const std::string& message, int mpi_code = 1) {
    std::ostringstream out;
    out << "{\"schema\":\"mpi_transport_probe_error_v1\",\"status\":\"failed\",\"rank\":"
        << world_rank << ",\"phase\":" << quote(phase) << ",\"error\":" << quote(message) << "}";
    std::cerr << out.str() << std::endl;
    if (world_rank == 0) std::cout << out.str() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, mpi_code ? mpi_code : 1);
    std::abort();
}

static void check(int code, const char* expression) {
    if (code == MPI_SUCCESS) return;
    char text[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(code, text, &length);
    fail(std::string(expression) + ": " + std::string(text, length), code);
}
#define MPICHECK(x) check((x), #x)

static unsigned char byte_value(int rank, int index, int iteration, int pattern) {
    return static_cast<unsigned char>((std::uint64_t(index) * 37 + rank * 101 + iteration * 17 + pattern * 53) % 251);
}
static float float_value(int rank, int index, int iteration, int pattern) {
    return float(int((std::uint64_t(index) * 37 + rank * 101 + iteration * 17 + pattern * 53) % 65521) - 32760) * 0.25f;
}
static std::uint64_t checksum(const void* data, std::size_t bytes) {
    std::uint64_t h = 14695981039346656037ULL;
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}
static std::string hex64(std::uint64_t x) {
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << x;
    return out.str();
}

struct Pattern { const char* name; bool floats; int count; int variation; bool zero_some; };

template<class T>
static std::string run_case(const Pattern& pattern, int pattern_index, int iteration, int size) {
    const int root = iteration % 2 == 0 ? 0 : size - 1;
    std::vector<int> counts(size), offsets(size);
    std::size_t total = 0;
    for (int rank = 0; rank < size; ++rank) {
        counts[rank] = pattern.zero_some && rank % 2 == 0 ? 0 : pattern.count + rank * pattern.variation;
        if (total + std::size_t(counts[rank]) > std::size_t(std::numeric_limits<int>::max())) fail("MPI count overflow");
        offsets[rank] = int(total);
        total += counts[rank];
    }
    std::vector<T> send(std::max(1, counts[world_rank]));
    for (int i = 0; i < counts[world_rank]; ++i)
        send[i] = pattern.floats ? T(float_value(world_rank, i, iteration, pattern_index)) : T(byte_value(world_rank, i, iteration, pattern_index));
    std::vector<T> receive(world_rank == root ? std::max<std::size_t>(1, total) : 0);
    // Mirrors the application's separate host send/receive buffers, including
    // ignored nullptr recvbuf on non-root and nonuniform counts/displacements.
    phase = std::string(pattern.name) + "/iteration" + std::to_string(iteration) + "/Gatherv";
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    MPICHECK(MPI_Gatherv(send.data(), counts[world_rank], pattern.floats ? MPI_FLOAT : MPI_BYTE,
                        world_rank == root ? receive.data() : nullptr, counts.data(), offsets.data(),
                        pattern.floats ? MPI_FLOAT : MPI_BYTE, root, MPI_COMM_WORLD));
    const double elapsed = MPI_Wtime() - start;
    int valid = 1;
    std::uint64_t actual_hash = 0;
    if (world_rank == root) {
        for (int rank = 0; rank < size; ++rank)
            for (int i = 0; i < counts[rank]; ++i) {
                const T expected = pattern.floats ? T(float_value(rank, i, iteration, pattern_index)) : T(byte_value(rank, i, iteration, pattern_index));
                if (receive[offsets[rank] + i] != expected) valid = 0;
            }
        actual_hash = checksum(receive.data(), total * sizeof(T));
    }
    phase += "/validation";
    MPICHECK(MPI_Bcast(&valid, 1, MPI_INT, root, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&actual_hash, 1, MPI_UINT64_T, root, MPI_COMM_WORLD));
    if (!valid) fail("received elements differ from deterministic rank/index pattern");
    double maximum_elapsed = 0;
    MPICHECK(MPI_Reduce(&elapsed, &maximum_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    std::ostringstream out;
    out << std::setprecision(17) << "{\"pattern\":" << quote(pattern.name) << ",\"iteration\":" << iteration
        << ",\"gatherv_root\":" << root << ",\"datatype\":" << quote(pattern.floats ? "MPI_FLOAT" : "MPI_BYTE")
        << ",\"counts\":[";
    for (int r = 0; r < size; ++r) out << (r ? "," : "") << counts[r];
    out << "],\"element_bytes\":" << sizeof(T) << ",\"received_bytes\":" << total * sizeof(T)
        << ",\"checksum_fnv1a64\":" << quote(hex64(actual_hash)) << ",\"elementwise_verified\":true"
        << ",\"max_rank_elapsed_progress_s\":" << maximum_elapsed << "}";
    return out.str();
}

int main(int argc, char** argv) {
    int provided = 0;
    int rc = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (rc != MPI_SUCCESS) return 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    try {
        int iterations = 4;
        std::string output;
        bool help = false;
        for (int i = 1; i < argc; ++i) {
            std::string arg(argv[i]);
            if (arg == "--help") help = true;
            else if (arg.rfind("--iterations=", 0) == 0) {
                const std::string value = arg.substr(13);
                std::size_t end = 0;
                iterations = std::stoi(value, &end);
                if (end != value.size()) fail("invalid iterations");
            } else if (arg.rfind("--output=", 0) == 0) output = arg.substr(9);
            else fail("unknown argument: " + arg);
        }
        if (iterations < 1 || iterations > 32) fail("iterations must be within1..32");
        if (help) {
            if (world_rank == 0) std::cout << "CPU MPI Gatherv probe: --iterations=1..32 --output=/local/existing-parent/report.json\n";
            MPI_Finalize();
            return 0;
        }
        int size = 0;
        MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
        if (size > 64) fail("diagnostic limited to64 ranks");
        MPI_Comm shared;
        phase = "topology";
        MPICHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &shared));
        int local_rank = 0, local_size = 0, leader = world_rank;
        MPICHECK(MPI_Comm_rank(shared, &local_rank));
        MPICHECK(MPI_Comm_size(shared, &local_size));
        MPICHECK(MPI_Bcast(&leader, 1, MPI_INT, 0, shared));
        int local_info[3] = {local_rank, local_size, leader};
        std::vector<int> info(world_rank == 0 ? size * 3 : 0);
        MPICHECK(MPI_Gather(local_info, 3, MPI_INT, info.data(), 3, MPI_INT, 0, MPI_COMM_WORLD));
        char hostname[MPI_MAX_PROCESSOR_NAME] = {};
        int hostname_length = 0;
        MPICHECK(MPI_Get_processor_name(hostname, &hostname_length));
        std::vector<char> hostnames(world_rank == 0 ? size * MPI_MAX_PROCESSOR_NAME : 0);
        MPICHECK(MPI_Gather(hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, hostnames.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD));
        char library[MPI_MAX_LIBRARY_VERSION_STRING] = {};
        int library_length = 0;
        MPICHECK(MPI_Get_library_version(library, &library_length));
        std::vector<std::string> results;
        auto report = [&](const char* status) {
            std::ostringstream out;
            out << "{\"schema\":\"cpu_mpi_transport_probe_v1\",\"status\":" << quote(status)
                << ",\"gpu_used\":false,\"continuous_monitoring\":false,\"performance_claim\":false"
                << ",\"ranks\":" << size << ",\"iterations\":" << iterations << ",\"mpi_library\":"
                << quote(std::string(library, library_length)) << ",\"rank_inventory\":[";
            for (int r = 0; r < size; ++r) {
                const char* h = hostnames.data() + r * MPI_MAX_PROCESSOR_NAME;
                std::size_t length = 0;
                while (length < MPI_MAX_PROCESSOR_NAME && h[length]) ++length;
                out << (r ? "," : "") << "{\"rank\":" << r << ",\"hostname\":" << quote(std::string(h, length))
                    << ",\"shared_local_rank\":" << info[3*r] << ",\"shared_local_size\":" << info[3*r+1]
                    << ",\"shared_leader_world_rank\":" << info[3*r+2] << "}";
            }
            out << "],\"cases\":[";
            for (std::size_t i = 0; i < results.size(); ++i) out << (i ? "," : "") << results[i];
            out << "],\"qualification\":\"Host buffers only; timings are diagnostic progress records. External timeout required. MPI errors may leave last checkpoint running and emit per-rank JSON errors to stderr.\"}";
            if (!output.empty()) {
                std::ofstream file(output);
                if (!file || !(file << out.str() << '\n')) fail("cannot write output: " + output);
            }
            if (std::string(status) == "completed") std::cout << out.str() << std::endl;
        };
        const Pattern patterns[] = {
            {"float_equal_1", true, 1, 0, false},
            {"float_equal_1024", true, 1024, 0, false},
            {"float_equal_7680_reproduces_pixel_dump", true, 7680, 0, false},
            {"float_heterogeneous_7680", true, 7680, 137, false},
            {"float_zero_and_uneven", true, 7680, 17, true},
            {"byte_equal_30720", false, 30720, 0, false},
            {"byte_heterogeneous_65536", false, 65536, 113, false}
        };
        if (world_rank == 0) report("running");
        for (int iteration = 0; iteration < iterations; ++iteration) {
            for (int p = 0; p < int(sizeof(patterns) / sizeof(patterns[0])); ++p) {
                const auto& pattern = patterns[p];
                std::string result = pattern.floats ? run_case<float>(pattern, p, iteration, size)
                                                    : run_case<unsigned char>(pattern, p, iteration, size);
                if (world_rank == 0) { results.push_back(std::move(result)); report("running"); }
            }
        }
        phase = "finalize";
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        if (world_rank == 0) report("completed");
        MPICHECK(MPI_Comm_free(&shared));
        MPICHECK(MPI_Finalize());
        return 0;
    } catch (const std::exception& error) { fail(error.what()); }
    catch (...) { fail("unknown exception"); }
}
