#include "basalt/models/qwen4exp/decode/dflash2.hpp"

#include "basalt/core/on_device.hpp"
#include "basalt/models/qwen4exp/decode/native_head.hpp"
#include "basalt/models/qwen4exp/kernels/q80_vhead.hpp"
#include "basalt/models/qwen4exp/kernels/q6k_head.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/spec/draft_policy.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace basalt::core {
namespace {

Dflash2* active = nullptr;

constexpr int N = 2560, NL = 5, NH = 32, NKV = 8, HD = 128;
constexpr int NFF = 7680, GROUPS = N / 16, KERNEL = 2;
constexpr int BLOCK = Dflash2::kBlock, TOPK = 16, RANK = 256, WINDOW = 2048, MASK_ID = 248077;
static_assert(Dflash2::kInject >= BLOCK && Dflash2::kInject >= Dflash2::kVerifyRows);
constexpr int ATTN_CHUNK = 256, ATTN_CHUNKS = WINDOW / ATTN_CHUNK, ATTN_PART = HD + 2;
constexpr float EPS = 1e-6f, THETA = 10000.0f;
constexpr float kAttnScale = 0.08838834764831845f;   // 1/sqrt(128)

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// x[rows][N] combined with base[side][tap][N] + dyn[rows][side][tap][group], causal taps inside the block.
__global__ void k_conv(const float* __restrict__ x, const float* __restrict__ dyn,
                       const float* __restrict__ base, float* __restrict__ out, int rows, int side) {
    const int t = blockIdx.x;
    const int e = threadIdx.x + blockIdx.y * blockDim.x;
    if (e >= N) return;
    const int group = e / 16;
    float acc = 0.0f;
    for (int tap = 0; tap < KERNEL; ++tap) {
        const int src = t - tap;
        if (src < 0) continue;
        const float w = dyn[(t * 2 + side) * (KERNEL * GROUPS) + tap * GROUPS + group] + base[(side * KERNEL + tap) * N + e];
        acc += w * x[src * N + e];
    }
    out[t * N + e] = acc;
}

// per-head RMSNorm then NeoX rope at absolute positions (one thread per head).
__global__ void k_norm_rope(float* __restrict__ x, const float* __restrict__ w, int rows, int heads,
                            int64_t pos0, const int32_t* staged_pos) {
    const int row = blockIdx.x, head = blockIdx.y;
    if (row >= rows || head >= heads) return;
    float* v = x + ((size_t) row * heads + head) * HD;
    float ss = 0.0f;
    for (int d = 0; d < HD; ++d) ss += v[d] * v[d];
    const float rs = rsqrtf(ss / (float) HD + EPS);
    for (int d = 0; d < HD; ++d) v[d] = v[d] * rs * w[d];
    const float p = (float) ((staged_pos ? (int64_t) *staged_pos : pos0) + row);
    for (int i = 0; i < HD / 2; ++i) {
        const float ang = p * __powf(THETA, -2.0f * (float) i / (float) HD);
        const float c = cosf(ang), s = sinf(ang);
        const float a = v[i], b = v[i + HD / 2];
        v[i] = a * c - b * s;
        v[i + HD / 2] = b * c + a * s;
    }
}

// One warp per (query row, head, ring chunk), with an independent online softmax.
__global__ void k_attn_part(const float* __restrict__ q, const float* __restrict__ kb, const float* __restrict__ vb,
                       const __half* __restrict__ kring, const __half* __restrict__ vring,
                       const int32_t* __restrict__ ringpos, float* __restrict__ partial, const int32_t* qpos0) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;   // 0 .. BLOCK*NH-1
    const int lane = threadIdx.x & 31;
    if (warp >= BLOCK * NH) return;
    const int row = warp / NH, head = warp % NH;
    const int kvh = head / (NH / NKV);
    const int64_t qi = (int64_t) *qpos0 + row;
    const float* qv = q + ((size_t) row * NH + head) * HD;
    float qreg[4];
    for (int i = 0; i < 4; ++i) qreg[i] = qv[lane * 4 + i];
    float m = -1e30f, l = 0.0f;
    float acc[4] = {0, 0, 0, 0};
    auto score = [&](const auto* k) {
        float s = 0.0f;
        for (int i = 0; i < 4; ++i) s += qreg[i] * (float) k[lane * 4 + i];
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        return s * kAttnScale;
    };
    auto take = [&](const auto* k, const auto* v) {
        const float sc = score(k);
        const float mn = fmaxf(m, sc);
        const float e = __expf(sc - mn), scale = __expf(m - mn);
        l = l * scale + e;
        for (int i = 0; i < 4; ++i) acc[i] = acc[i] * scale + e * (float) v[lane * 4 + i];
        m = mn;
    };
    const int chunk = blockIdx.y;
    if (chunk == 0) {
        for (int kk = 0; kk < BLOCK; ++kk) {
            take(kb + ((size_t) kk * NKV + kvh) * HD, vb + ((size_t) kk * NKV + kvh) * HD);
        }
    }
    for (int c = chunk * ATTN_CHUNK; c < (chunk + 1) * ATTN_CHUNK && c < qi; ++c) {
        const int64_t kp = ringpos[c];
        if (kp < 0 || kp >= qi || qi - kp >= WINDOW) continue;
        take(kring + ((size_t) c * NKV + kvh) * HD, vring + ((size_t) c * NKV + kvh) * HD);
    }
    float* part = partial + ((size_t) chunk * BLOCK * NH + warp) * ATTN_PART;
    for (int i = 0; i < 4; ++i) part[lane * 4 + i] = acc[i];
    if (lane == 0) {
        part[HD] = m;
        part[HD + 1] = l;
    }
}

__global__ void k_attn_merge(const float* __restrict__ partial, float* __restrict__ out) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int dim = (threadIdx.x & 31) * 4;
    if (warp >= BLOCK * NH) return;

    float max_score = -1e30f;
    for (int chunk = 0; chunk < ATTN_CHUNKS; ++chunk) {
        const float* part = partial + ((size_t) chunk * BLOCK * NH + warp) * ATTN_PART;
        max_score = fmaxf(max_score, part[HD]);
    }
    float sum = 0.0f, acc[4] = {};
    for (int chunk = 0; chunk < ATTN_CHUNKS; ++chunk) {
        const float* part = partial + ((size_t) chunk * BLOCK * NH + warp) * ATTN_PART;
        const float weight = __expf(part[HD] - max_score);
        sum += weight * part[HD + 1];
        for (int i = 0; i < 4; ++i) acc[i] += weight * part[dim + i];
    }
    for (int i = 0; i < 4; ++i) out[(size_t) warp * HD + dim + i] = acc[i] / sum;
}

__global__ void k_silu_mul(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    out[i] = (g / (1.0f + __expf(-g))) * up[i];
}

__global__ void k_add(float* __restrict__ x, const float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

constexpr int kTopTiles = 16, kTopThreads = 128;

__device__ void insert_top(float value, int32_t id, float* values, int32_t* ids) {
    auto better = [&](int at) { return value > values[at] || (value == values[at] && id < ids[at]); };
    if (id < 0 || !better(TOPK - 1)) return;
    int at = TOPK - 1;
    while (at > 0 && better(at - 1)) {
        values[at] = values[at - 1];
        ids[at] = ids[at - 1];
        --at;
    }
    values[at] = value;
    ids[at] = id;
}

__global__ void top_tiles(const float* logits, int vocab, int32_t* ids, float* values) {
    const int row = blockIdx.x;
    const int tile = blockIdx.y;
    const int start = vocab * tile / kTopTiles, end = vocab * (tile + 1) / kTopTiles;
    const float* source = logits + (size_t) row * vocab;
    __shared__ float local_values[kTopThreads][TOPK];
    __shared__ int32_t local_ids[kTopThreads][TOPK];
    float best[TOPK];
    int32_t tokens[TOPK];
    for (int i = 0; i < TOPK; ++i) { best[i] = -INFINITY; tokens[i] = -1; }
    for (int i = start + threadIdx.x; i < end; i += kTopThreads)
        insert_top(source[i], i, best, tokens);
    for (int i = 0; i < TOPK; ++i) {
        local_values[threadIdx.x][i] = best[i];
        local_ids[threadIdx.x][i] = tokens[i];
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    for (int i = 0; i < TOPK; ++i) { best[i] = -INFINITY; tokens[i] = -1; }
    for (int i = 0; i < kTopThreads; ++i)
        for (int j = 0; j < TOPK; ++j)
            insert_top(local_values[i][j], local_ids[i][j], best, tokens);
    const int offset = (row * kTopTiles + tile) * TOPK;
    for (int i = 0; i < TOPK; ++i) { values[offset + i] = best[i]; ids[offset + i] = tokens[i]; }
}

__global__ void merge_top(const int32_t* tile_ids, const float* tile_values, int32_t* ids, float* values, int rows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const int offset = row * kTopTiles * TOPK;
    float best[TOPK];
    int32_t tokens[TOPK];
    for (int i = 0; i < TOPK; ++i) { best[i] = -INFINITY; tokens[i] = -1; }
    for (int i = 0; i < kTopTiles * TOPK; ++i)
        insert_top(tile_values[offset + i], tile_ids[offset + i], best, tokens);
    for (int i = 0; i < TOPK; ++i) { values[row * TOPK + i] = best[i]; ids[row * TOPK + i] = tokens[i]; }
}

// Mirror the verifier's Philox draw on the host without advancing or altering its sampler.
float proposal_uniform(uint64_t seed, uint64_t pos) {
    uint32_t c0 = (uint32_t) pos, c1 = (uint32_t) (pos >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    for (uint32_t i = 0; i < 10; ++i) {
        const uint64_t p0 = uint64_t{0x9e3779b9u} * c0;
        const uint64_t p1 = uint64_t{0xbb67ae85u} * c2;
        c0 = (uint32_t) (p1 >> 32) ^ c1 ^ i;
        c1 = (uint32_t) p1;
        c2 = (uint32_t) (p0 >> 32) ^ c3;
        c3 = (uint32_t) p0;
    }
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}

int coupled_pick(const float* scores, const int32_t* ids, const kernels::SamplerParams& sp, uint64_t pos) {
    int order[TOPK];
    for (int i = 0; i < TOPK; ++i) order[i] = i;
    std::sort(order, order + TOPK, [&](int a, int b) {
        return scores[a] > scores[b] || (scores[a] == scores[b] && ids[a] < ids[b]);
    });
    int n = kernels::sampler_list_len(sp.top_k, TOPK);
    const float peak = scores[order[0]];
    if (sp.top_p < 1.0f) {
        double total = 0.0;
        for (int i = 0; i < n; ++i) total += std::exp((double) scores[order[i]] - peak);
        double prefix = 0.0;
        for (int i = 0; i < n; ++i) {
            prefix += std::exp((double) scores[order[i]] - peak);
            if (i + 1 >= sp.min_keep && prefix >= sp.top_p * total) { n = i + 1; break; }
        }
    }
    if (sp.min_p > 0.0f) {
        const float cut = peak + std::log(sp.min_p);
        while (n > 1 && scores[order[n - 1]] < cut) --n;
    }
    double masses[TOPK], total = 0.0;
    for (int i = 0; i < n; ++i) {
        masses[i] = std::exp(((double) scores[order[i]] - peak) / sp.temperature);
        total += masses[i];
    }
    double draw = proposal_uniform(sp.seed, pos) * total;
    for (int i = 0; i < n - 1; ++i) {
        if (draw < masses[i]) return order[i];
        draw -= masses[i];
    }
    return order[n - 1];
}

float bf16_to_f32(uint16_t b) { const uint32_t u = (uint32_t) b << 16; float f; std::memcpy(&f, &u, 4); return f; }

// The producer stores the mean of the four completed hyper-connection streams.
__global__ void capture_features(float* dst, const float* residual, const float* out,
                                 const float* inject, int tap, int64_t pos, bool ring) {
    const int row = blockIdx.x;
    const int channel = blockIdx.y * blockDim.x + threadIdx.x;
    if (channel >= N) return;
    const size_t i = (size_t) row * 4 * N + channel;
    double sum = 0;
    for (int h = 0; h < 4; ++h) {
        float x = residual[i + h * N];
        if (out != nullptr) x = fmaf(out[(size_t) row * N + channel], inject[row * 4 + h], x);
        sum += x;
    }
    const size_t cell = ring ? (pos + row) % WINDOW : row;
    dst[(cell * NL + tap) * N + channel] = (float) (sum * 0.25);
}

__global__ void store_ring(__half* k_ring, __half* v_ring, int32_t* positions, const float* k, const float* v,
                           int64_t pos) {
    const int row = blockIdx.x;
    const int channel = blockIdx.y * blockDim.x + threadIdx.x;
    const int i = row * NKV * HD + channel;
    const int cell = (pos + row) % WINDOW;
    k_ring[(size_t) cell * NKV * HD + channel] = __float2half_rn(k[i]);
    v_ring[(size_t) cell * NKV * HD + channel] = __float2half_rn(v[i]);
    if (channel == 0) positions[cell] = (int32_t) (pos + row);
}

int tap_index(int64_t layer) {
    return layer >= 5 && layer <= 45 && (layer - 5) % 10 == 0 ? (int) (layer - 5) / 10 : -1;
}

}  // namespace

Dflash2* active_dflash2() { return active; }

Dflash2::~Dflash2() { release(); }

void Dflash2::release() {
    if (active == this) active = nullptr;
    if (dev_ < 0) return;
    int previous = 0;
    (void) cudaGetDevice(&previous);
    (void) cudaSetDevice(dev_);
    if (stream_) { cudaStreamSynchronize((cudaStream_t) stream_); cudaStreamDestroy((cudaStream_t) stream_); stream_ = nullptr; }
    if (round_graph_) { cudaGraphExecDestroy(round_graph_); round_graph_ = nullptr; }
    if (round_io_) { cudaFreeHost(round_io_); round_io_ = nullptr; }
    if (profile_rounds_) {
        std::fprintf(stderr, "basalt dflash2 timing: rounds=%llu inject=%.3fms block=%.3fms head=%.3fms topk_io=%.3fms widths=",
                     (unsigned long long) profile_rounds_, profile_ms_[0] / profile_rounds_,
                     profile_ms_[1] / profile_rounds_, profile_ms_[2] / profile_rounds_, profile_ms_[3] / profile_rounds_);
        for (int i = 0; i <= kDrafts; ++i) {
            std::fprintf(stderr, "%s%llu", i ? "," : "", (unsigned long long) profile_proposals_[i]);
        }
        std::fprintf(stderr, "\n");
    }
    for (auto& event : profile_events_) {
        if (event) { cudaEventDestroy(event); event = nullptr; }
    }
    graph_enabled_ = false;
    round_warmed_ = false;
    head_ = nullptr;
    if (dense_) { cudaFree(dense_); dense_ = nullptr; }
    if (arena_) { cudaFree(arena_); arena_ = nullptr; }
    if (prompt_) { cudaFreeHost(prompt_); prompt_ = nullptr; }
    (void) cudaSetDevice(previous);
}

const Dflash2::Tensor* Dflash2::find(const std::string& n) const {
    for (const auto& t : tensors_) if (t.name == n) return &t;
    return nullptr;
}
const void* Dflash2::q4(const char* n) const {
    const Tensor* t = find(n);
    return (t == nullptr || t->kind != "q4_0") ? nullptr : dense_ + t->off;
}
const float* Dflash2::f32(const char* n) const {
    const Tensor* t = find(n);
    return (t == nullptr || t->kind != "f32") ? nullptr : (const float*) (dense_ + t->off);
}
float Dflash2::cb_prev(int64_t tok, int k) const { return bf16_to_f32(prev_cb_[(size_t) tok * RANK + k]); }
float Dflash2::cb_succ(int64_t tok, int k) const { return bf16_to_f32(succ_cb_[(size_t) tok * RANK + k]); }

bool Dflash2::load(const std::string& path, const NativeHead& head, const NativeEmbed* embed,
                   int device, std::string& err) {
    if (active || loaded() || !head.loaded() || embed == nullptr) {
        err = "dflash2: needs the target's loaded native head and embedding";
        return false;
    }
    dev_ = device;
    n_vocab_ = 248320;
    embed_ = embed;
    try {
        const OnDevice on(dev_);
        const std::string& rt_dir = path;
        std::ifstream idx(rt_dir + "/dense.txt");
        if (!idx) { err = "dflash2: cannot open " + rt_dir + "/dense.txt (run tools/dflash2_pack.py)"; return false; }
        std::string line;
        while (std::getline(idx, line)) {
            if (line.empty()) continue;
            std::istringstream is(line);
            Tensor t; is >> t.name >> t.kind >> t.rows >> t.cols >> t.off >> t.bytes;
            if (!is || (t.kind != "q4_0" && t.kind != "f32" && t.kind != "bf16")) {
                err = "dflash2: malformed dense.txt line: " + line;
                return false;
            }
            tensors_.push_back(t);
        }
        std::ifstream bin(rt_dir + "/dense.bin", std::ios::binary);
        if (!bin) { err = "dflash2: cannot open dense.bin"; return false; }
        bin.seekg(0, std::ios::end);
        const uint64_t file_bytes = (uint64_t) bin.tellg();
        uint64_t gpu_bytes = 0;
        for (const auto& t : tensors_) {
            if (t.off > file_bytes || t.bytes > file_bytes - t.off) {
                err = "dflash2: tensor extends past dense.bin: " + t.name;
                return false;
            }
            if (t.kind != "bf16") gpu_bytes = std::max(gpu_bytes, t.off + t.bytes);
        }
        const auto read_book = [&](const char* name, std::vector<uint16_t>& book) {
            const Tensor* t = find(name);
            if (!t || t->kind != "bf16" || t->off < gpu_bytes || t->bytes != (size_t) n_vocab_ * RANK * 2) return false;
            book.resize(t->bytes / 2);
            bin.seekg((std::streamoff) t->off);
            return (bool) bin.read((char*) book.data(), (std::streamsize) t->bytes);
        };
        if (!gpu_bytes ||
            !read_book("candidate_selector.predecessor_codebook", prev_cb_) ||
            !read_book("candidate_selector.successor_codebook", succ_cb_)) {
            err = "dflash2: missing GPU weights or host selector codebooks";
            return false;
        }
        ck(cudaMalloc((void**) &dense_, gpu_bytes), "dflash2: weights allocation");
        {
            std::vector<uint8_t> blob((size_t) gpu_bytes);
            bin.seekg(0);
            if (!bin.read((char*) blob.data(), (std::streamsize) gpu_bytes)) {
                err = "dflash2: reading dense.bin failed";
                return false;
            }
            ck(cudaMemcpy(dense_, blob.data(), gpu_bytes, cudaMemcpyHostToDevice), "dflash2: weights upload");
        }
        const char* need[] = {"fc.weight", "hidden_norm.weight", "norm.weight",
                              "candidate_selector.hidden_projection.weight"};
        for (const char* n : need) {
            if (!q4(n) && !f32(n)) { err = std::string("dflash2: missing tensor ") + n; return false; }
        }
        cudaStream_t cs = nullptr;
        ck(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "dflash2: stream");
        stream_ = cs;
        const uint64_t ring = (uint64_t) NL * WINDOW * NKV * HD * sizeof(__half);
        const uint64_t parts[] = {
            ring, ring, (uint64_t) NL * WINDOW * sizeof(int32_t),
            (uint64_t) BLOCK * N * 4, (uint64_t) BLOCK * N * 4, (uint64_t) BLOCK * (2 * KERNEL * GROUPS) * 4,
            (uint64_t) BLOCK * NH * HD * 4, (uint64_t) kInject * NKV * HD * 4, (uint64_t) kInject * NKV * HD * 4,
            (uint64_t) BLOCK * (NH * HD) * 4, (uint64_t) BLOCK * N * 4,
            (uint64_t) BLOCK * NFF * 4, (uint64_t) BLOCK * NFF * 4, (uint64_t) BLOCK * N * 4,
            (uint64_t) kInject * NL * N * 4, (uint64_t) kInject * N * 4,
            (uint64_t) kDrafts * n_vocab_ * 4, (uint64_t) kDrafts * RANK * 4,
            (uint64_t) basalt::kernels::native_q8_1_bytes((int) (NL * N), kInject),
            (uint64_t) basalt::kernels::native_q8_1_bytes((int) NFF, BLOCK),
            (uint64_t) kDrafts * TOPK * 4, (uint64_t) kDrafts * TOPK * 4,
            (uint64_t) kVerifyRows * NL * N * 4, (uint64_t) (BLOCK + 1) * sizeof(int32_t),
            (uint64_t) kDrafts * kTopTiles * TOPK * sizeof(int32_t),
            (uint64_t) kDrafts * kTopTiles * TOPK * sizeof(float),
            head.vproxy() ? kernels::q6k_vhead_scratch_bytes() : 0,
            (uint64_t) ATTN_CHUNKS * BLOCK * NH * ATTN_PART * sizeof(float),
        };
        uint64_t sum = 0;
        for (uint64_t p : parts) sum += (p + 15) & ~15ull;
        arena_bytes_ = sum;
        ck(cudaMalloc(&arena_, arena_bytes_), "dflash2: workspace allocation");
        uint8_t* p = (uint8_t*) arena_;
        auto take = [&](uint64_t bytes) { uint8_t* q = p; p += (bytes + 15) & ~15ull; return q; };
        kring_ = (__half*) take(parts[0]); vring_ = (__half*) take(parts[1]); kpos_ = (int32_t*) take(parts[2]);
        x_ = (float*) take(parts[3]); h_ = (float*) take(parts[4]); dyn_ = (float*) take(parts[5]);
        q_ = (float*) take(parts[6]); k_ = (float*) take(parts[7]); v_ = (float*) take(parts[8]);
        att_ = (float*) take(parts[9]); o_ = (float*) take(parts[10]);
        gate_ = (float*) take(parts[11]); up_ = (float*) take(parts[12]); y_ = (float*) take(parts[13]);
        feat_ = (float*) take(parts[14]); g_ = (float*) take(parts[15]);
        logits_ = (float*) take(parts[16]); gates_ = (float*) take(parts[17]);
        xq_ = (uint8_t*) take(parts[18]); xq2_ = (uint8_t*) take(parts[19]);
        top_ids_ = (int32_t*) take(parts[20]); top_vals_ = (float*) take(parts[21]);
        verify_ = (float*) take(parts[22]);
        tokens_ = (int32_t*) take(parts[23]);
        top_tile_ids_ = (int32_t*) take(parts[24]);
        top_tile_vals_ = (float*) take(parts[25]);
        if (parts[26]) {
            vhead_scratch_ = take(parts[26]);
            ck(cudaMemsetAsync(vhead_scratch_, 0, parts[26], cs), "dflash2: head scratch init");
        }
        ck(cudaHostAlloc((void**) &round_io_, sizeof(RoundIo), cudaHostAllocDefault), "dflash2: draft staging");
        const char* graph = std::getenv("BASALT_DFLASH_GRAPH");
        graph_enabled_ = graph == nullptr || std::strcmp(graph, "0") != 0;
        if (const char* conf = std::getenv("BASALT_DFLASH_CONF")) {
            char* end = nullptr;
            const float value = std::strtof(conf, &end);
            if (end == conf || *end != '\0' || !(value > 0.0f && value <= 1.0f))
                throw std::runtime_error("BASALT_DFLASH_CONF must be in (0, 1]");
            max_selector_sum_ = 1.0f / value;
        }
        const char* coupled = std::getenv("BASALT_DFLASH_COUPLED");
        coupled_proposals_ = coupled && std::strcmp(coupled, "1") == 0;
        if (const char* profile = std::getenv("BASALT_DFLASH_PROFILE"); profile && std::strcmp(profile, "1") == 0) {
            for (auto& event : profile_events_) ck(cudaEventCreate(&event), "dflash2: profile event");
        }
        att_part_ = (float*) take(parts[27]);
        const size_t prompt_bytes = (size_t) WINDOW * NL * N * sizeof(float);
        ck(cudaHostAlloc((void**) &prompt_, prompt_bytes, cudaHostAllocMapped | cudaHostAllocPortable), "dflash2: prompt staging");
        ck(cudaHostGetDevicePointer((void**) &prompt_device_, prompt_, 0), "dflash2: prompt mapping");
        ck(cudaMemsetAsync(kpos_, 0xff, (size_t) NL * WINDOW * sizeof(int32_t), cs), "dflash2: ring init");
        ck(cudaStreamSynchronize(cs), "dflash2: ring init sync");
        head_ = &head;
        active = this;
        std::fprintf(stderr, "basalt dflash2: Q4_0 %llu bytes, ring+workspace %llu bytes on CUDA%d; "
                             "shared target head; host staging and selector codebooks\n",
                     (unsigned long long) gpu_bytes, (unsigned long long) arena_bytes_, dev_);
    } catch (const std::exception& e) {
        err = e.what();
        release();
        return false;
    }
    return true;
}

// Also used by the decode block below.
void dflash2_norm_rope(float* x, const float* w, int rows, int heads, int64_t pos0, void* stream,
                      const int32_t* staged_pos = nullptr);

bool Dflash2::begin_prompt(int64_t read_from, int64_t prompt_size, bool live_prefix, std::string& err) {
    prompt_end_ = std::max<int64_t>(0, prompt_size - 1);
    prompt_first_ = std::max<int64_t>(read_from, prompt_size - WINDOW);
    seed_pending_ = true;
    if (live_prefix && ring_end_ == read_from) return true;
    ring_end_ = -1;
    try {
        const OnDevice on(dev_);
        const auto cs = (cudaStream_t) stream_;
        ck(cudaMemsetAsync(kpos_, 0xff, (size_t) NL * WINDOW * sizeof(int32_t), cs), "dflash2: reset ring");
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

void Dflash2::capture_prefill(int64_t layer, const float* residual, int rows, int64_t pos, cudaStream_t stream) const {
    const int tap = tap_index(layer);
    if (tap < 0 || pos + rows <= prompt_first_ || pos >= prompt_end_) return;
    const int skip = (int) std::max<int64_t>(0, prompt_first_ - pos);
    const int count = (int) std::min<int64_t>(rows - skip, prompt_end_ - pos - skip);
    if (count <= 0) return;
    capture_features<<<dim3(count, N / 256), 256, 0, stream>>>(prompt_device_, residual + (size_t) skip * 4 * N,
                                                               nullptr, nullptr, tap, pos + skip, true);
}

void Dflash2::capture_verify(int64_t layer, const float* residual, const float* out, const float* inject,
                             bool pending, int rows, cudaStream_t stream) const {
    const int tap = tap_index(layer);
    if (tap < 0 || rows > kVerifyRows) return;
    capture_features<<<dim3(rows, N / 256), 256, 0, stream>>>(verify_, residual, pending ? out : nullptr,
                                                              inject, tap, 0, false);
}

bool Dflash2::remember_verify(int64_t pos, int rows, std::string& err) {
    const int skip = (int) std::min<int64_t>(rows, std::max<int64_t>(0, prompt_first_ - pos));
    const int end = (int) std::max<int64_t>(skip, std::min<int64_t>(rows, prompt_end_ - pos));
    if (skip == end) return true;
    try {
        const OnDevice on(dev_);
        const auto cs = (cudaStream_t) stream_;
        for (int i = skip; i < end;) {
            const int cell = (pos + i) % WINDOW;
            const int count = std::min(end - i, WINDOW - cell);
            ck(cudaMemcpyAsync(prompt_ + (size_t) cell * NL * N, verify_ + (size_t) i * NL * N,
                               (size_t) count * NL * N * sizeof(float), cudaMemcpyDeviceToHost, cs), "dflash2: prompt rows");
            i += count;
        }
        ck(cudaStreamSynchronize(cs), "dflash2: prompt rows complete");
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

void Dflash2::inject_rows(const float* rows, int64_t pos, int count) {
    const auto cs = (cudaStream_t) stream_;
    kernels::native_quantize_q8_1(rows, xq_, NL * N, count, cs);
    kernels::native_mmvq(2, q4("fc.weight"), xq_, g_, NL * N, N, count, cs);
    kernels::rms_norm_weighted(g_, f32("hidden_norm.weight"), count, N, EPS, cs);
    kernels::native_quantize_q8_1(g_, xq_, N, count, cs);
    for (int l = 0; l < NL; ++l) {
        char name[96];
        std::snprintf(name, sizeof name, "layers.%d.self_attn.k_proj.weight", l);
        kernels::native_mmvq(2, q4(name), xq_, k_, N, NKV * HD, count, cs);
        std::snprintf(name, sizeof name, "layers.%d.self_attn.v_proj.weight", l);
        kernels::native_mmvq(2, q4(name), xq_, v_, N, NKV * HD, count, cs);
        std::snprintf(name, sizeof name, "layers.%d.self_attn.k_norm.weight", l);
        dflash2_norm_rope(k_, f32(name), count, NKV, pos, cs);
        const size_t off = (size_t) l * WINDOW * NKV * HD;
        store_ring<<<dim3(count, NKV * HD / 256), 256, 0, cs>>>(kring_ + off, vring_ + off,
                                                                kpos_ + l * WINDOW, k_, v_, pos);
    }
}

bool Dflash2::seed_prompt(int64_t end, std::string& err) {
    if (!seed_pending_) return true;
    seed_pending_ = false;
    try {
        const OnDevice on(dev_);
        const auto cs = (cudaStream_t) stream_;
        end = std::min(end, prompt_end_);
        for (int64_t pos = prompt_first_; pos < end;) {
            const int cell = pos % WINDOW;
            const int count = (int) std::min<int64_t>({kInject, end - pos, WINDOW - cell});
            ck(cudaMemcpyAsync(feat_, prompt_ + (size_t) cell * NL * N, (size_t) count * NL * N * sizeof(float),
                               cudaMemcpyHostToDevice, cs), "dflash2: inject staging");
            inject_rows(feat_, pos, count);
            pos += count;
        }
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

bool Dflash2::inject_verified(int64_t pos, int rows, std::string& err) {
    try {
        const OnDevice on(dev_);
        mark_profile(0);
        inject_rows(verify_, pos, rows);
        ring_end_ = pos + rows;
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

bool Dflash2::finish(std::string& err) {
    const OnDevice on(dev_);
    const auto status = cudaStreamSynchronize((cudaStream_t) stream_);
    if (status == cudaSuccess) return true;
    ring_end_ = -1;
    err = std::string("dflash2: finish: ") + cudaGetErrorString(status);
    return false;
}

void Dflash2::forward_block() {
    const auto cs = (cudaStream_t) stream_;
    ck(cudaMemcpyAsync(tokens_, round_io_->tokens, sizeof(round_io_->tokens), cudaMemcpyHostToDevice, cs),
       "dflash2: draft inputs");
    embed_->gather_dev(tokens_, BLOCK, x_, cs);
    for (int l = 0; l < NL; ++l) {
        char nm[96];
        ck(cudaMemcpyAsync(h_, x_, (size_t) BLOCK * N * sizeof(float), cudaMemcpyDeviceToDevice, cs), "dflash2: h=x");
        std::snprintf(nm, sizeof nm, "layers.%d.input_layernorm.weight", l);
        basalt::kernels::rms_norm_weighted(h_, f32(nm), BLOCK, N, EPS, cs);
        std::snprintf(nm, sizeof nm, "layers.%d.attention_conv.kernel_projection.weight", l);
        basalt::kernels::native_quantize_q8_1(h_, xq_, N, BLOCK, cs);
        basalt::kernels::native_mmvq(2, q4(nm), xq_, dyn_, N, 2 * KERNEL * GROUPS, BLOCK, cs);
        std::snprintf(nm, sizeof nm, "layers.%d.attention_conv.base_kernel", l);
        const float* abase = f32(nm);
        {
            dim3 g(BLOCK, (N + 255) / 256);
            k_conv<<<g, 256, 0, cs>>>(h_, dyn_, abase, o_, BLOCK, 0);
        }
        ck(cudaMemcpyAsync(h_, o_, (size_t) BLOCK * N * sizeof(float), cudaMemcpyDeviceToDevice, cs), "dflash2: h=convA");
        basalt::kernels::native_quantize_q8_1(h_, xq_, N, BLOCK, cs);
        std::snprintf(nm, sizeof nm, "layers.%d.self_attn.q_proj.weight", l);
        basalt::kernels::native_mmvq(2, q4(nm), xq_, q_, N, NH * HD, BLOCK, cs);
        std::snprintf(nm, sizeof nm, "layers.%d.self_attn.k_proj.weight", l);
        basalt::kernels::native_mmvq(2, q4(nm), xq_, k_, N, NKV * HD, BLOCK, cs);
        std::snprintf(nm, sizeof nm, "layers.%d.self_attn.v_proj.weight", l);
        basalt::kernels::native_mmvq(2, q4(nm), xq_, v_, N, NKV * HD, BLOCK, cs);
        std::snprintf(nm, sizeof nm, "layers.%d.self_attn.q_norm.weight", l);
        dflash2_norm_rope(q_, f32(nm), BLOCK, NH, 0, cs, tokens_ + BLOCK);
        std::snprintf(nm, sizeof nm, "layers.%d.self_attn.k_norm.weight", l);
        dflash2_norm_rope(k_, f32(nm), BLOCK, NKV, 0, cs, tokens_ + BLOCK);
        {
            const int warps = BLOCK * NH;
            const int blocks = (warps * 32 + 255) / 256;
            const size_t off = (size_t) l * WINDOW * NKV * HD;
            k_attn_part<<<dim3(blocks, ATTN_CHUNKS), 256, 0, cs>>>(
                q_, k_, v_, kring_ + off, vring_ + off, kpos_ + (size_t) l * WINDOW, att_part_, tokens_ + BLOCK);
            k_attn_merge<<<blocks, 256, 0, cs>>>(att_part_, att_);
        }
        basalt::kernels::native_quantize_q8_1(att_, xq_, NH * HD, BLOCK, cs);
        std::snprintf(nm, sizeof nm, "layers.%d.self_attn.o_proj.weight", l);
        basalt::kernels::native_mmvq(2, q4(nm), xq_, o_, NH * HD, N, BLOCK, cs);
        {
            dim3 g(BLOCK, (N + 255) / 256);
            k_conv<<<g, 256, 0, cs>>>(o_, dyn_, abase, h_, BLOCK, 1);
        }
        k_add<<<(BLOCK * N + 255) / 256, 256, 0, cs>>>(x_, h_, BLOCK * N);
        // FFN
        ck(cudaMemcpyAsync(h_, x_, (size_t) BLOCK * N * sizeof(float), cudaMemcpyDeviceToDevice, cs), "dflash2: h=x2");
        std::snprintf(nm, sizeof nm, "layers.%d.post_attention_layernorm.weight", l);
        basalt::kernels::rms_norm_weighted(h_, f32(nm), BLOCK, N, EPS, cs);
        std::snprintf(nm, sizeof nm, "layers.%d.mlp_conv.kernel_projection.weight", l);
        basalt::kernels::native_quantize_q8_1(h_, xq_, N, BLOCK, cs);
        basalt::kernels::native_mmvq(2, q4(nm), xq_, dyn_, N, 2 * KERNEL * GROUPS, BLOCK, cs);
        std::snprintf(nm, sizeof nm, "layers.%d.mlp_conv.base_kernel", l);
        const float* fbase = f32(nm);
        {
            dim3 g(BLOCK, (N + 255) / 256);
            k_conv<<<g, 256, 0, cs>>>(h_, dyn_, fbase, o_, BLOCK, 0);
        }
        ck(cudaMemcpyAsync(h_, o_, (size_t) BLOCK * N * sizeof(float), cudaMemcpyDeviceToDevice, cs), "dflash2: h=convF");
        basalt::kernels::native_quantize_q8_1(h_, xq_, N, BLOCK, cs);
        std::snprintf(nm, sizeof nm, "layers.%d.mlp.gate_proj.weight", l);
        basalt::kernels::native_mmvq(2, q4(nm), xq_, gate_, N, NFF, BLOCK, cs);
        std::snprintf(nm, sizeof nm, "layers.%d.mlp.up_proj.weight", l);
        basalt::kernels::native_mmvq(2, q4(nm), xq_, up_, N, NFF, BLOCK, cs);
        k_silu_mul<<<(BLOCK * NFF + 255) / 256, 256, 0, cs>>>(gate_, up_, gate_, BLOCK * NFF);
        basalt::kernels::native_quantize_q8_1(gate_, xq2_, NFF, BLOCK, cs);
        std::snprintf(nm, sizeof nm, "layers.%d.mlp.down_proj.weight", l);
        basalt::kernels::native_mmvq(2, q4(nm), xq2_, y_, NFF, N, BLOCK, cs);
        {
            dim3 g(BLOCK, (N + 255) / 256);
            k_conv<<<g, 256, 0, cs>>>(y_, dyn_, fbase, h_, BLOCK, 1);
        }
        k_add<<<(BLOCK * N + 255) / 256, 256, 0, cs>>>(x_, h_, BLOCK * N);
    }
    ck(cudaMemcpyAsync(h_, x_, (size_t) BLOCK * N * sizeof(float), cudaMemcpyDeviceToDevice, cs), "dflash2: h=final");
    basalt::kernels::rms_norm_weighted(h_, f32("norm.weight"), BLOCK, N, EPS, cs);
}

void dflash2_norm_rope(float* x, const float* w, int rows, int heads, int64_t pos0, void* stream,
                      const int32_t* staged_pos) {
    dim3 g(rows, heads);
    k_norm_rope<<<g, 1, 0, (cudaStream_t) stream>>>(x, w, rows, heads, pos0, staged_pos);
}

void Dflash2::mark_profile(int stage, bool capturing) {
    if (const auto event = profile_events_[stage]) {
        ck(cudaEventRecordWithFlags(event, (cudaStream_t) stream_, capturing ? cudaEventRecordExternal : 0),
           "dflash2: profile record");
    }
}

void Dflash2::enqueue_round(int count, bool capturing) {
    const auto cs = (cudaStream_t) stream_;
    forward_block();
    mark_profile(2, capturing);
    // The anchor and unused tail only condition the block. Their logits are not consumed.
    kernels::native_quantize_q8_1(h_ + N, xq_, N, count, cs);
    kernels::native_mmvq(2, q4("candidate_selector.hidden_projection.weight"), xq_, gates_, N, RANK, count, cs);
    if (vhead_scratch_ && kernels::q6k_vhead_supported(N, (int) n_vocab_, count)) {
        kernels::q6k_vhead(head_->vproxy(), head_->weights(), xq_, logits_, N, (int) n_vocab_, count, vhead_scratch_, cs);
    } else if (head_->type() == 14 && kernels::q6k_head_supported(N, count)) {
        kernels::q6k_head_mmvq(head_->weights(), xq_, logits_, N, (int) n_vocab_, count, cs);
    } else if (head_->planes()) {
        kernels::q80p_mmvq(head_->weights(), xq_, logits_, N, (int) n_vocab_, count, cs);
    } else {
        kernels::native_mmvq(head_->type(), head_->weights(), xq_, logits_, N, (int) n_vocab_, count, cs);
    }
    mark_profile(3, capturing);
    top_tiles<<<dim3(count, kTopTiles), kTopThreads, 0, cs>>>(logits_, (int) n_vocab_, top_tile_ids_, top_tile_vals_);
    merge_top<<<1, 32, 0, cs>>>(top_tile_ids_, top_tile_vals_, top_ids_, top_vals_, count);
    ck(cudaMemcpyAsync(round_io_->ids, top_ids_, count * TOPK * sizeof(int32_t), cudaMemcpyDeviceToHost, cs), "dflash2: ids");
    ck(cudaMemcpyAsync(round_io_->vals, top_vals_, count * TOPK * sizeof(float), cudaMemcpyDeviceToHost, cs), "dflash2: vals");
    ck(cudaMemcpyAsync(round_io_->gates, gates_, count * RANK * sizeof(float), cudaMemcpyDeviceToHost, cs), "dflash2: gates");
}

void Dflash2::launch_round(int count) {
    const auto cs = (cudaStream_t) stream_;
    if (graph_enabled_ && count == kDrafts) {
        if (round_warmed_ && !round_graph_) {
            cudaGraph_t graph = nullptr;
            auto status = cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal);
            if (status == cudaSuccess) {
                try {
                    enqueue_round(count, true);
                } catch (...) {
                    cudaStreamEndCapture(cs, &graph);
                    if (graph) cudaGraphDestroy(graph);
                    graph_enabled_ = false;
                    throw;
                }
                status = cudaStreamEndCapture(cs, &graph);
                if (status == cudaSuccess) status = cudaGraphInstantiate(&round_graph_, graph, 0);
                if (graph) cudaGraphDestroy(graph);
            }
            if (status == cudaSuccess) {
                std::fprintf(stderr, "basalt dflash2: captured full draft round\n");
            } else {
                graph_enabled_ = false;
                std::fprintf(stderr, "basalt dflash2: using direct launches after capture failure: %s\n",
                             cudaGetErrorString(status));
            }
        }
        if (round_graph_) {
            ck(cudaGraphLaunch(round_graph_, cs), "dflash2: graph launch");
            return;
        }
        round_warmed_ = true;
    }
    enqueue_round(count);
}

bool Dflash2::draft(int32_t anchor, int64_t pos, int max_drafts, const kernels::SamplerParams& sampling,
                    const spec::DraftPolicy* cost_policy, int32_t* out, int& n_out, std::string& err,
                    void (*after_enqueue)(void*), void* enqueue_context) {
    n_out = 0;
    try {
        int caller_dev = dev_;
        if (after_enqueue) ck(cudaGetDevice(&caller_dev), "dflash2: commit device");
        const OnDevice on(dev_);
        const int count = std::min(kDrafts, max_drafts);
        if (count <= 0) return finish(err);
        round_io_->tokens[0] = anchor;
        std::fill_n(round_io_->tokens + 1, BLOCK - 1, MASK_ID);
        round_io_->tokens[BLOCK] = (int32_t) pos;
        mark_profile(1);
        launch_round(count);
        mark_profile(4);
        if (after_enqueue) {
            const OnDevice on_caller(caller_dev);
            after_enqueue(enqueue_context);
        }
        ck(cudaStreamSynchronize((cudaStream_t) stream_), "dflash2: draft sync");
        int32_t pred = anchor;
        int best_width = 0;
        double expected = 1.0, prefix_hit = 1.0;
        double best_rate = cost_policy ? 1.0 / cost_policy->cost_ms(1) : 0.0;
        for (int i = 0; i < count; ++i) {
            const int32_t* cand = round_io_->ids + (size_t) i * TOPK;
            const float* uni = round_io_->vals + (size_t) i * TOPK;
            const float* gate = round_io_->gates + (size_t) i * RANK;
            float best = -1e30f;
            int bk = 0;
            float scores[TOPK];
            for (int k = 0; k < TOPK; ++k) {
                float s = uni[k];
                const int64_t tok = cand[k];
                if (tok < 0 || tok >= n_vocab_) { err = "dflash2: invalid selector token"; return false; }
                for (int r = 0; r < RANK; ++r) s += cb_succ(tok, r) * (cb_prev(pred, r) * gate[r]);
                scores[k] = s;
                if (s > best) { best = s; bk = k; }
            }
            float sum = 0.0f;
            for (const float s : scores) sum += std::exp(s - best);
            if (!std::isfinite(sum) || (!cost_policy && sum > max_selector_sum_)) break;
            if (coupled_proposals_ && !sampling.greedy && sampling.temperature > 0.0f)
                bk = coupled_pick(scores, cand, sampling, (uint64_t) pos + i);
            pred = cand[bk];
            out[n_out++] = pred;
            if (cost_policy) {
                // Selector confidence is optimistic for an exact sampled target match.
                prefix_hit *= 0.70 / sum;
                expected += prefix_hit;
                const double rate = expected / cost_policy->cost_ms(n_out + 1);
                if (rate > best_rate) { best_rate = rate; best_width = n_out; }
            }
        }
        if (cost_policy) n_out = best_width;
        if (profile_events_[0]) {
            for (int i = 0; i < kProfileMarks - 1; ++i) {
                float ms = 0.0f;
                ck(cudaEventElapsedTime(&ms, profile_events_[i], profile_events_[i + 1]), "dflash2: profile elapsed");
                profile_ms_[i] += ms;
            }
            ++profile_rounds_;
            ++profile_proposals_[n_out];
        }
        return true;
    } catch (const std::exception& e) {
        ring_end_ = -1;
        err = e.what();
        return false;
    }
}

}  // namespace basalt::core
