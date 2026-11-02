// Microbenchmark of the delta-net decode step (AVX2 path of iqk_fused_delta_net_impl, head_dim 128):
// why is splitting a head's value rows over two threads slower?
//   dnbench MODE CPU_A [CPU_B]
// MODE: full  = one thread, whole heads, rows [0,128) as a compile-time loop (current kernel)
//       var   = one thread, whole heads, rows [r0,r1) runtime bounds (patched kernel, nsplit 1)
//       halfv = two threads, each half of every head, runtime bounds (patched kernel, nsplit 2)
//       halfc = two threads, each half of every head, compile-time 64-row loops
//       solo  = one thread, half of every head only (runtime bounds), the other half untouched
// The state set is 30 layers x 16 heads x 64 KiB (one node's heads), cycled so it is cold like at decode.
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

constexpr int HD = 128, NH = 16, NL = 30;
static float * g_state;            // [NL][NH][HD*HD]
static float g_q[HD], g_k[HD], g_v[HD], g_out[NL][NH][HD];

template <int R0, int R1>
static inline void head_const(float * state, float * out, int r_off) {
    float v_prime[HD], out_val[HD], v_new_buf[HD];
    const float scale = 1.0f / sqrtf((float)HD), decay = 0.97f, beta_val = 0.6f;
    float kq = 0; for (int i = 0; i < HD; ++i) kq += g_k[i]*g_q[i];
    const float attn_score = kq*scale;
    float * st = state + r_off;
    std::memset(v_prime, 0, sizeof v_prime); std::memset(out_val, 0, sizeof out_val);
    for (int col = 0; col < HD; ++col) {
        const float k_col = g_k[col], q_col = g_q[col];
        for (int row = R0; row < R1; ++row) { const float s = st[row + col*HD]; v_prime[row] += s*k_col; out_val[row] += s*q_col; }
    }
    for (int row = R0; row < R1; ++row) {
        const float v_new = g_v[row+r_off]*beta_val - v_prime[row]*beta_val*decay;
        v_new_buf[row] = v_new; out[row+r_off] = out_val[row]*decay*scale + v_new*attn_score;
    }
    auto vd = _mm256_set1_ps(decay), vmin = _mm256_set1_ps(-1e6f), vmax = _mm256_set1_ps(1e6f);
    for (int col = 0; col < HD; ++col) {
        auto vk = _mm256_set1_ps(g_k[col]);
        for (int row = R0; row < R1; row += 8) {
            auto vs = _mm256_loadu_ps(st + col*HD + row); auto vn = _mm256_loadu_ps(v_new_buf + row);
            vs = _mm256_fmadd_ps(vn, vk, _mm256_mul_ps(vs, vd));
            vs = _mm256_max_ps(vmin, _mm256_min_ps(vmax, vs));
            _mm256_storeu_ps(st + col*HD + row, vs);
        }
    }
}

__attribute__((noinline)) static void head_var(float * state, float * out, int r0, int r1) {
    float v_prime[HD], out_val[HD], v_new_buf[HD];
    const float scale = 1.0f / sqrtf((float)HD), decay = 0.97f, beta_val = 0.6f;
    float kq = 0; for (int i = 0; i < HD; ++i) kq += g_k[i]*g_q[i];
    const float attn_score = kq*scale;
    std::memset(v_prime, 0, sizeof v_prime); std::memset(out_val, 0, sizeof out_val);
    for (int col = 0; col < HD; ++col) {
        const float k_col = g_k[col], q_col = g_q[col];
        for (int row = r0; row < r1; ++row) { const float s = state[row + col*HD]; v_prime[row] += s*k_col; out_val[row] += s*q_col; }
    }
    for (int row = r0; row < r1; ++row) {
        const float v_new = g_v[row]*beta_val - v_prime[row]*beta_val*decay;
        v_new_buf[row] = v_new; out[row] = out_val[row]*decay*scale + v_new*attn_score;
    }
    auto vd = _mm256_set1_ps(decay), vmin = _mm256_set1_ps(-1e6f), vmax = _mm256_set1_ps(1e6f);
    for (int col = 0; col < HD; ++col) {
        auto vk = _mm256_set1_ps(g_k[col]);
        for (int row = r0; row < r1; row += 8) {
            auto vs = _mm256_loadu_ps(state + col*HD + row); auto vn = _mm256_loadu_ps(v_new_buf + row);
            vs = _mm256_fmadd_ps(vn, vk, _mm256_mul_ps(vs, vd));
            vs = _mm256_max_ps(vmin, _mm256_min_ps(vmax, vs));
            _mm256_storeu_ps(state + col*HD + row, vs);
        }
    }
}

static const char * g_mode;
static std::atomic<int> g_go{0}, g_done{0};
static constexpr int ITERS = 40;   // passes over the whole state set
static double g_ns[2];

static void pin(int cpu) { cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s); pthread_setaffinity_np(pthread_self(), sizeof s, &s); }

struct Arg { int cpu, part; };
static void * worker(void * p) {
    Arg a = *(Arg *)p; pin(a.cpu);
    const bool full = !strcmp(g_mode, "full"), var = !strcmp(g_mode, "var"), halfc = !strcmp(g_mode, "halfc");
    g_go.fetch_add(1); while (g_go.load() < 100) {}
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < ITERS; ++it) {
        for (int l = 0; l < NL; ++l) {
            for (int h = 0; h < NH; ++h) {
                float * st = g_state + ((size_t)l*NH + h)*HD*HD;
                float * out = g_out[l][h];
                if (full) head_const<0, HD>(st, out, 0);
                else if (var) head_var(st, out, 0, HD);
                else if (halfc) head_const<0, HD/2>(st, out, a.part*HD/2);
                else head_var(st, out, a.part*HD/2, a.part*HD/2 + HD/2);
            }
            // both threads finish a layer before the next (one barrier per layer, as in the graph)
            if (!full && !var && strcmp(g_mode, "solo")) {
                int target = (it*NL + l + 1)*2;
                g_done.fetch_add(1); while (g_done.load() < target) _mm_pause();
            }
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    g_ns[a.part] = std::chrono::duration<double, std::nano>(t1 - t0).count();
    return nullptr;
}

int main(int argc, char ** argv) {
    g_mode = argv[1];
    size_t n = (size_t)NL*NH*HD*HD;
    // DNB_OFFSET=<bytes>: shift the state off cache-line alignment (ggml only guarantees 32 B)
    const size_t off = getenv("DNB_OFFSET") ? atoi(getenv("DNB_OFFSET")) : 0;
    g_state = (float *)((char *)aligned_alloc(2*1024*1024, n*sizeof(float) + 4096) + off);
    for (size_t i = 0; i < n; ++i) g_state[i] = (float)((i*2654435761u) % 1000) * 1e-4f;
    for (int i = 0; i < HD; ++i) { g_q[i] = 0.01f*(i%7); g_k[i] = 0.02f*(i%5); g_v[i] = 0.03f*(i%3); }
    const bool two = !strcmp(g_mode, "halfv") || !strcmp(g_mode, "halfc");
    pthread_t th[2]; Arg args[2] = {{atoi(argv[2]), 0}, {two ? atoi(argv[3]) : 0, 1}};
    pthread_create(&th[0], nullptr, worker, &args[0]);
    if (two) pthread_create(&th[1], nullptr, worker, &args[1]);
    while (g_go.load() < (two ? 2 : 1)) {}
    g_go.store(100);
    pthread_join(th[0], nullptr); if (two) pthread_join(th[1], nullptr);
    double per_layer_us = g_ns[0] / (ITERS*NL) / 1000.0;
    printf("%-6s cpus %s %s  per layer (16 heads): %.1f us  (per head-pass %.2f us)\n", g_mode, argv[2], two ? argv[3] : "-",
           per_layer_us, per_layer_us/NH);
    return 0;
}
