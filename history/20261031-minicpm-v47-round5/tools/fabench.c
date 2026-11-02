// Round-5 flash-attention decode microbenchmark for the MiniCPM-V 4.7 (qwen35moe) shape:
// 16 Q heads, 2 KV heads, head dim 256, F16 K/V, single token, F16 mask.
// Calls iqk_flash_attn_noalibi from libggml exactly the way ggml_compute_forward_flash_attn_ext_f16 does.
//
//   fabench <n_ctx_used> <nth> [layers=20] [reps=20]
// Pin threads with GOMP_CPU_AFFINITY / OMP_PROC_BIND. Distinct K/V per layer so KV streams from DRAM.
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>

typedef void (*barrier_t)(void *);
bool iqk_flash_attn_noalibi(int type_q, int type_mask, float max_bias,
                            int neq3, int neq2, long nbq3, long nbq2,
                            int nek3, int nek2, long nbk3, long nbk2,
                            int nev3, int nev2, long nbv3, long nbv2,
                            int ne2, int ne1, long nb1,
                            int type_k, int type_v, int Dk, int Dv, int nq, int nk,
                            int stride_q, int stride_k, int stride_v, int stride_m,
                            const void * q, const void * k, const void * v, const void * mask, const void * sinks,
                            float scale, float softcap, float * qkv,
                            void * work_buffer, barrier_t barrier, void * barrier_data,
                            int ith, int nth, int n_swa, void * indexer);

enum { D = 256, NQH = 16, NKVH = 2, TYPE_F32 = 0, TYPE_F16 = 1 };

static void omp_bar(void * unused) { (void)unused;
    #pragma omp barrier
}

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9*t.tv_nsec; }

static uint16_t f2h(float f) { return _cvtss_sh(f, 0); }
static float h2f(uint16_t h) { return _cvtsh_ss(h); }

static void * huge_alloc(size_t n) { n = (n + (2u<<20) - 1) & ~(size_t)((2u<<20) - 1); void * p = aligned_alloc(2u<<20, n); madvise(p, n, MADV_HUGEPAGE); return p; }

static uint64_t rng = 88172645463325252ull;
static float frand(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (float)((rng >> 11)*(1.0/9007199254740992.0)) * 2.0f - 1.0f; }

int main(int argc, char ** argv) {
    const int n_used = argc > 1 ? atoi(argv[1]) : 2834;
    const int nth    = argc > 2 ? atoi(argv[2]) : 32;
    const int layers = argc > 3 ? atoi(argv[3]) : 20;
    const int reps   = argc > 4 ? atoi(argv[4]) : 20;
    const int n_kv   = (n_used + 255)/256*256;              // llama pads n_kv to 256
    const long row   = (long)NKVH*D*2;                       // bytes per KV position (both heads)
    const float scale = 1.0f/sqrtf((float)D);

    uint16_t ** K = malloc(sizeof(*K)*layers), ** V = malloc(sizeof(*V)*layers);
    for (int l = 0; l < layers; ++l) {
        // KV_STRIDE_KB=S: K and V of a layer in one buffer with V = K + S KiB (mimics ik's
        // 16384-position cache tensors, which put V exactly 16 MiB after K); default: separate buffers
        const char * se = getenv("KV_STRIDE_KB");
        if (se) {
            const size_t stride = (size_t)atol(se) << 10;
            char * b = huge_alloc(stride + row*n_kv);
            K[l] = (uint16_t *)b; V[l] = (uint16_t *)(b + stride);
        } else {
            K[l] = huge_alloc(row*n_kv); V[l] = huge_alloc(row*n_kv);
        }
        for (long i = 0; i < (long)n_kv*NKVH*D; ++i) { K[l][i] = f2h(frand()); V[l][i] = f2h(frand()); }
        // INIT_PAR=1: rewrite every KV row from a different thread (row-interleaved over all pinned
        // threads), like prompt processing does with its batch thread pool
        if (getenv("INIT_PAR") && atoi(getenv("INIT_PAR")) > 0) {
            uint16_t * Kl = K[l], * Vl = V[l];
            #pragma omp parallel for schedule(static, 1) num_threads(nth)
            for (long j = 0; j < n_kv; ++j) {
                for (int d = 0; d < NKVH*D; ++d) { Kl[j*NKVH*D + d] ^= 0; Vl[j*NKVH*D + d] ^= 0; }
                __asm__ volatile("" ::: "memory");
                memcpy(Kl + j*NKVH*D, Kl + j*NKVH*D, 0);
                volatile uint16_t * kv = Kl + j*NKVH*D; kv[0] = kv[0];
                volatile uint16_t * vv = Vl + j*NKVH*D; vv[0] = vv[0];
                for (int d = 0; d < NKVH*D; d += 32) { ((volatile uint16_t *)Kl)[j*NKVH*D + d] = Kl[j*NKVH*D + d]; ((volatile uint16_t *)Vl)[j*NKVH*D + d] = Vl[j*NKVH*D + d]; }
            }
        }
    }
    float * Q = aligned_alloc(64, sizeof(float)*NQH*D);
    for (int i = 0; i < NQH*D; ++i) Q[i] = 2.0f*frand();
    uint16_t * M = aligned_alloc(64, 2*(size_t)n_kv);
    for (int i = 0; i < n_kv; ++i) M[i] = f2h(i < n_used ? 0.0f : -INFINITY);
    float * out = aligned_alloc(64, sizeof(float)*NQH*D);
    void * work = aligned_alloc(64, 64 << 20);

    const char * fe = getenv("FLUSH_MB");
    const size_t flush_n = fe ? (size_t)atol(fe) << 20 : 0;
    char * flush = flush_n ? huge_alloc(flush_n) : NULL;
    const int flush_write = getenv("FLUSH_WRITE") && atoi(getenv("FLUSH_WRITE")) > 0;
    if (flush) memset(flush, 1, flush_n);
    volatile long sink = 0;
    double best = 1e30, sum = 0, fsum = 0; int cnt = 0;
    for (int r = 0; r < reps; ++r) {
        double t0 = now();
        #pragma omp parallel num_threads(nth)
        {
            const int ith = omp_get_thread_num();
            for (int l = 0; l < layers; ++l) {
                iqk_flash_attn_noalibi(TYPE_F32, TYPE_F16, 0.0f,
                    1, NQH, (long)NQH*D*4, (long)D*4,
                    1, NKVH, row*n_kv, (long)D*2,
                    1, NKVH, row*n_kv, (long)D*2,
                    1, NQH, (long)D*4,
                    TYPE_F16, TYPE_F16, D, D, 1, n_kv,
                    NQH*D*4, (int)row, (int)row, n_kv*2,
                    Q, K[l], V[l], M, NULL, scale, 0.0f, out,
                    work, omp_bar, NULL, ith, nth, 0, NULL);
                #pragma omp barrier
                if (flush) {
                    double tf = ith == 0 ? now() : 0;
                    const size_t part = flush_n/nth;
                    const __m256i * fp = (const __m256i *)(flush + ith*part);
                    __m256i acc = _mm256_setzero_si256();
                    if (flush_write) {
                        // dirty the lines (like the per-token recurrent-state / activation writes)
                        __m256i * wp = (__m256i *)fp;
                        const __m256i v = _mm256_set1_epi32(ith + 1);
                        for (size_t i = 0; i < part/32; ++i) _mm256_store_si256(wp + i, v);
                    } else {
                        for (size_t i = 0; i < part/32; ++i) acc = _mm256_xor_si256(acc, _mm256_load_si256(fp + i));
                    }
                    sink += _mm256_extract_epi64(acc, 0);
                    #pragma omp barrier
                    if (ith == 0) fsum += now() - tf;
                }
            }
        }
        double dt = (now() - t0 - fsum)/layers; fsum = 0;
        if (r > 0) { sum += dt; cnt++; }
        if (dt < best) best = dt;
    }
    const double bytes = 2.0*row*n_used;
    printf("n_used %d n_kv %d nth %d: mean %.1f us/layer, best %.1f us/layer, %.1f GB/s (KV bytes / mean)\n",
           n_used, n_kv, nth, 1e6*sum/cnt, 1e6*best, bytes/(sum/cnt)/1e9);

    // reference check on the last layer
    const int l = layers - 1;
    double maxerr = 0, maxref = 0;
    float * s = malloc(sizeof(float)*n_used);
    for (int h = 0; h < NQH; ++h) {
        const int kh = h/(NQH/NKVH);
        float mx = -INFINITY;
        for (int j = 0; j < n_used; ++j) {
            double acc = 0;
            for (int d = 0; d < D; ++d) acc += (double)Q[h*D + d]*h2f(K[l][(long)j*NKVH*D + kh*D + d]);
            s[j] = (float)(acc*scale); if (s[j] > mx) mx = s[j];
        }
        double S = 0; for (int j = 0; j < n_used; ++j) { s[j] = expf(s[j] - mx); S += s[j]; }
        for (int d = 0; d < D; ++d) {
            double acc = 0;
            for (int j = 0; j < n_used; ++j) acc += s[j]*h2f(V[l][(long)j*NKVH*D + kh*D + d]);
            acc /= S;
            double e = fabs(acc - out[h*D + d]); if (e > maxerr) maxerr = e;
            if (fabs(acc) > maxref) maxref = fabs(acc);
        }
    }
    printf("max abs err vs double reference %.3g (max |ref| %.3g)\n", maxerr, maxref);
    return 0;
}
