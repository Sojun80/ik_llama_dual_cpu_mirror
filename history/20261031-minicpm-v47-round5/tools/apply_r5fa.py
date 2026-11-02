#!/usr/bin/env python3
"""Round-5 patch 3: dedicated single-token GQA flash-attention decode kernel (env IK_R5_FA=1).

Shape: F32 Q, F16 K/V, F16 mask, head dim 256, rk2 = neq2/nek2 a multiple of 4 (<= 16),
one query token, no softcap / sinks / SWA / indexer. Anything else uses ik's path.

Each thread owns a contiguous range of KV positions for *all* KV heads, so it streams
whole K/V rows. Per block of 32 positions:
  scores  : 4-head x 2-position register tile, K converted f16->f32 once per tile
  softmax : online (running max / sum), ik's v_expf
  V accum : 4 heads x 16 dims of O held in registers across the block
Partials (m, l, O) are merged with log-sum-exp weights after one barrier.
"""
import sys, pathlib

root = pathlib.Path(sys.argv[1])
fa = root / 'ggml/src/iqk/iqk_flash_attn.cpp'
s = fa.read_text()
assert 'r5fa' not in s

KERNEL = r'''
// ---- round-5 decode flash attention (env IK_R5_FA=1) ------------------------------------
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#include "iqk_utils.h"
#include <immintrin.h>
#include <cstdlib>
namespace r5fa {
constexpr int D  = 256;
constexpr int JB = 32;
constexpr int kMaxHeads = 16;

static bool enabled() {
    static const bool e = [] { auto s = getenv("IK_R5_FA"); return s && atoi(s) > 0; }();
    return e;
}

static inline size_t slot_floats(int neq2) { return (2*(size_t)neq2 + (size_t)neq2*D + 15) & ~(size_t)15; }

static inline bool eligible(int type_q, int type_k, int type_v, int type_mask, int Dk, int Dv,
                            int neq1, int neq2, int neq3, int nek2, int nek3, int nev2, int nev3,
                            float max_bias, float softcap, const void * mask, const void * sinks,
                            int n_swa, const void * indexer) {
    if (!enabled()) return false;
    if (type_q != GGML_TYPE_F32 || type_k != GGML_TYPE_F16 || type_v != GGML_TYPE_F16 || type_mask != GGML_TYPE_F16) return false;
    if (Dk != D || Dv != D || neq1 != 1 || neq3 != 1 || nek3 != 1 || nev3 != 1) return false;
    if (nek2 != nev2 || nek2 < 1 || neq2 % nek2 || neq2 > kMaxHeads) return false;
    const int rk = neq2/nek2;
    if (rk % 4 || rk > 16) return false;
    return max_bias <= 0.0f && softcap == 0.0f && mask && !sinks && n_swa == 0 && !indexer;
}

static inline float hsum(__m256 x) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_movehdup_ps(a));
    return _mm_cvtss_f32(a);
}

// partial attention of one thread over positions [j0, j1) for all heads.
// Block-outer / head-inner so each block touches whole K and V rows (all KV heads) once,
// and the next block's rows are software-prefetched while the current one is computed.
static void partial(int j0, int j1, int nkvh, int rk,
                    const char * q, long nbq2, const char * k, long stride_k, long nbk2,
                    const char * v, long stride_v, long nbv2, const uint16_t * mask, float scale,
                    float * m_out, float * l_out, float * O_all) {
    const int nh = nkvh*rk;                 // <= kMaxHeads
    alignas(32) float Qs[kMaxHeads*D];
    alignas(32) float S[kMaxHeads*JB];
    float m[kMaxHeads], l[kMaxHeads];
    const __m256 vs = _mm256_set1_ps(scale);
    for (int h = 0; h < nh; ++h) {
        const float * qh = (const float *)(q + h*nbq2);
        for (int d = 0; d < D; d += 8) _mm256_store_ps(Qs + h*D + d, _mm256_mul_ps(vs, _mm256_loadu_ps(qh + d)));
        m[h] = -INFINITY; l[h] = 0.0f;
    }
    std::memset(O_all, 0, sizeof(float)*nh*D);
    const long span_k = (nkvh - 1)*nbk2 + 2*D;   // bytes of one position K row over all KV heads
    const long span_v = (nkvh - 1)*nbv2 + 2*D;
    for (int jb = j0; jb < j1; jb += JB) {
        const int nj = std::min(JB, j1 - jb);
        const int jn = jb + JB, nn = std::max(0, std::min(JB, j1 - jn));
        // ---- scores S[h][j] = (scale*q_h) . k_j
        for (int kh = 0; kh < nkvh; ++kh) {
            const char * kbase = k + kh*nbk2;
            for (int j = 0; j < nj; j += 2) {
                if (kh == 0) {
                    for (int jj = j; jj < j + 2 && jj < nn; ++jj) {
                        const char * kr = k + (long)(jn + jj)*stride_k;
                        const char * vr = v + (long)(jn + jj)*stride_v;
                        for (long o = 0; o < span_k; o += 64) _mm_prefetch(kr + o, _MM_HINT_T1);
                        for (long o = 0; o < span_v; o += 64) _mm_prefetch(vr + o, _MM_HINT_T1);
                    }
                }
                const uint16_t * k0 = (const uint16_t *)(kbase + (long)(jb + j)*stride_k);
                const uint16_t * k1 = j + 1 < nj ? (const uint16_t *)(kbase + (long)(jb + j + 1)*stride_k) : k0;
                for (int h0 = kh*rk; h0 < (kh + 1)*rk; h0 += 4) {
                    __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps(), a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
                    __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps(), a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps();
                    const float * q0 = Qs + (h0 + 0)*D, * q1 = Qs + (h0 + 1)*D, * q2 = Qs + (h0 + 2)*D, * q3 = Qs + (h0 + 3)*D;
                    for (int d = 0; d < D; d += 8) {
                        const __m256 x0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k0 + d)));
                        const __m256 x1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k1 + d)));
                        __m256 y = _mm256_load_ps(q0 + d); a00 = _mm256_fmadd_ps(y, x0, a00); a01 = _mm256_fmadd_ps(y, x1, a01);
                        y = _mm256_load_ps(q1 + d);        a10 = _mm256_fmadd_ps(y, x0, a10); a11 = _mm256_fmadd_ps(y, x1, a11);
                        y = _mm256_load_ps(q2 + d);        a20 = _mm256_fmadd_ps(y, x0, a20); a21 = _mm256_fmadd_ps(y, x1, a21);
                        y = _mm256_load_ps(q3 + d);        a30 = _mm256_fmadd_ps(y, x0, a30); a31 = _mm256_fmadd_ps(y, x1, a31);
                    }
                    S[(h0 + 0)*JB + j] = hsum(a00); S[(h0 + 1)*JB + j] = hsum(a10);
                    S[(h0 + 2)*JB + j] = hsum(a20); S[(h0 + 3)*JB + j] = hsum(a30);
                    if (j + 1 < nj) {
                        S[(h0 + 0)*JB + j + 1] = hsum(a01); S[(h0 + 1)*JB + j + 1] = hsum(a11);
                        S[(h0 + 2)*JB + j + 1] = hsum(a21); S[(h0 + 3)*JB + j + 1] = hsum(a31);
                    }
                }
            }
        }
        // ---- mask + online softmax; S becomes p
        alignas(32) float mk[JB];
        for (int j = 0; j < JB; ++j) mk[j] = j < nj ? _cvtsh_ss(mask[jb + j]) : -INFINITY;
        for (int h = 0; h < nh; ++h) {
            float * sh = S + h*JB;
            float bm = -INFINITY;
            for (int j = 0; j < JB; ++j) { sh[j] = j < nj ? sh[j] + mk[j] : -INFINITY; bm = std::max(bm, sh[j]); }
            if (bm == -INFINITY) { std::memset(sh, 0, sizeof(float)*JB); continue; }
            if (bm > m[h]) {
                if (m[h] > -INFINITY) {
                    const float c = expf(m[h] - bm);
                    l[h] *= c;
                    const __m256 vc = _mm256_set1_ps(c);
                    float * oh = O_all + h*D;
                    for (int d = 0; d < D; d += 8) _mm256_storeu_ps(oh + d, _mm256_mul_ps(vc, _mm256_loadu_ps(oh + d)));
                }
                m[h] = bm;
            }
            const __m256 vm = _mm256_set1_ps(m[h]);
            __m256 vsum = _mm256_setzero_ps();
            for (int j = 0; j < JB; j += 8) {
                const __m256 p = v_expf(_mm256_sub_ps(_mm256_load_ps(sh + j), vm));
                _mm256_store_ps(sh + j, p);
                vsum = _mm256_add_ps(vsum, p);
            }
            l[h] += hsum(vsum);
        }
        // ---- O[h] += sum_j p[h][j] * v_j
        for (int kh = 0; kh < nkvh; ++kh) {
            const char * vbase = v + kh*nbv2;
            for (int h0 = kh*rk; h0 < (kh + 1)*rk; h0 += 4) {
                const float * p0 = S + (h0 + 0)*JB, * p1 = S + (h0 + 1)*JB, * p2 = S + (h0 + 2)*JB, * p3 = S + (h0 + 3)*JB;
                for (int d = 0; d < D; d += 16) {
                    float * o0 = O_all + (h0 + 0)*D + d, * o1 = O_all + (h0 + 1)*D + d, * o2 = O_all + (h0 + 2)*D + d, * o3 = O_all + (h0 + 3)*D + d;
                    __m256 c00 = _mm256_loadu_ps(o0), c01 = _mm256_loadu_ps(o0 + 8), c10 = _mm256_loadu_ps(o1), c11 = _mm256_loadu_ps(o1 + 8);
                    __m256 c20 = _mm256_loadu_ps(o2), c21 = _mm256_loadu_ps(o2 + 8), c30 = _mm256_loadu_ps(o3), c31 = _mm256_loadu_ps(o3 + 8);
                    for (int j = 0; j < nj; ++j) {
                        const uint16_t * vr = (const uint16_t *)(vbase + (long)(jb + j)*stride_v) + d;
                        const __m256 x0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)vr));
                        const __m256 x1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(vr + 8)));
                        __m256 w = _mm256_broadcast_ss(p0 + j); c00 = _mm256_fmadd_ps(w, x0, c00); c01 = _mm256_fmadd_ps(w, x1, c01);
                        w = _mm256_broadcast_ss(p1 + j);        c10 = _mm256_fmadd_ps(w, x0, c10); c11 = _mm256_fmadd_ps(w, x1, c11);
                        w = _mm256_broadcast_ss(p2 + j);        c20 = _mm256_fmadd_ps(w, x0, c20); c21 = _mm256_fmadd_ps(w, x1, c21);
                        w = _mm256_broadcast_ss(p3 + j);        c30 = _mm256_fmadd_ps(w, x0, c30); c31 = _mm256_fmadd_ps(w, x1, c31);
                    }
                    _mm256_storeu_ps(o0, c00); _mm256_storeu_ps(o0 + 8, c01); _mm256_storeu_ps(o1, c10); _mm256_storeu_ps(o1 + 8, c11);
                    _mm256_storeu_ps(o2, c20); _mm256_storeu_ps(o2 + 8, c21); _mm256_storeu_ps(o3, c30); _mm256_storeu_ps(o3 + 8, c31);
                }
            }
        }
    }
    for (int h = 0; h < nh; ++h) { m_out[h] = m[h]; l_out[h] = l[h]; }
}

static bool run(int neq2, long nbq2, int nek2, long nbk2, long nbv2, long nb1, int nek1,
                int stride_k, int stride_v, const void * q, const void * k, const void * v, const void * mask,
                float scale, float * qkv, void * work, barrier_t barrier, void * barrier_data, int ith, int nth) {
    const uint16_t * mk = (const uint16_t *)mask;
    // effective length: drop the trailing fully masked (-inf) positions
    int n_eff = nek1;
    while (n_eff > 0 && mk[n_eff - 1] == 0xFC00) --n_eff;
    const int rk = neq2/nek2;
    const size_t slot = slot_floats(neq2);
    float * mine = (float *)work + (size_t)ith*slot;
    // contiguous position ranges, multiples of 8
    const int per = ((n_eff + nth - 1)/nth + 7) & ~7;
    const int j0 = std::min(n_eff, ith*per), j1 = std::min(n_eff, j0 + per);
    static const bool dbg = [] { auto s = getenv("IK_R5_FA_DEBUG"); return s && atoi(s) > 0; }();
    const uint64_t t0 = dbg ? __rdtsc() : 0;
    partial(j0, j1, nek2, rk, (const char *)q, nbq2, (const char *)k, stride_k, nbk2, (const char *)v, stride_v, nbv2,
            mk, scale, mine, mine + neq2, mine + 2*neq2);
    const uint64_t t1 = dbg ? __rdtsc() : 0;
    barrier(barrier_data);
    if (dbg) {
        // per-thread accumulation of partial() ticks and barrier wait; thread 0 reports every 640 calls
        static uint64_t acc_part[256], acc_wait[256], ncall;
        acc_part[ith] += t1 - t0; acc_wait[ith] += __rdtsc() - t1;
        if (ith == 0 && ++ncall % 640 == 0) {
            uint64_t mx = 0, sm = 0, wsm = 0;
            for (int t = 0; t < nth; ++t) { mx = std::max(mx, acc_part[t]); sm += acc_part[t]; wsm += acc_part[t] ? acc_wait[t] : 0; }
            fprintf(stderr, "r5fa: calls %llu nek1 %d n_eff %d per %d stride_k %d nbk2 %ld stride_v %d nbv2 %ld nbq2 %ld nb1 %ld rk %d | "
                    "partial ticks/call max %.0f mean %.0f  wait mean %.0f | k %% 4096 = %ld\n",
                    (unsigned long long)ncall, nek1, n_eff, per, stride_k, nbk2, stride_v, nbv2, nbq2, nb1, rk,
                    (double)mx/640, (double)sm/nth/640, (double)wsm/nth/640, (long)((uintptr_t)k % 4096));
            for (int t = 0; t < nth; ++t) acc_part[t] = acc_wait[t] = 0;
        }
    }
    // merge: tasks are (head, 64-dim chunk)
    constexpr int DC = 64;
    const int ntask = neq2*(D/DC);
    for (int task = ith; task < ntask; task += nth) {
        const int h = task/(D/DC), d0 = (task%(D/DC))*DC;
        float M = -INFINITY;
        for (int t = 0; t < nth; ++t) M = std::max(M, ((const float *)work + t*slot)[h]);
        __m256 acc[DC/8];
        for (int i = 0; i < DC/8; ++i) acc[i] = _mm256_setzero_ps();
        float L = 0.0f;
        if (M > -INFINITY) {
            for (int t = 0; t < nth; ++t) {
                const float * pt = (const float *)work + t*slot;
                if (pt[h] == -INFINITY) continue;
                const float w = expf(pt[h] - M);
                L += w*pt[neq2 + h];
                const __m256 vw = _mm256_set1_ps(w);
                const float * Ot = pt + 2*neq2 + (size_t)h*D + d0;
                for (int i = 0; i < DC/8; ++i) acc[i] = _mm256_fmadd_ps(vw, _mm256_loadu_ps(Ot + 8*i), acc[i]);
            }
        }
        const __m256 vn = _mm256_set1_ps(L > 0 ? 1.0f/L : 0.0f);
        float * out = (float *)((char *)qkv + h*nb1) + d0;
        for (int i = 0; i < DC/8; ++i) _mm256_storeu_ps(out + 8*i, _mm256_mul_ps(vn, acc[i]));
    }
    return true;
}
} // namespace r5fa
#define R5FA 1
#endif
// ---- end round-5 decode flash attention -------------------------------------------------
'''

anchor = 'size_t iqk_fa_work_buffer_size(const struct ggml_tensor * dst, int nth) {'
assert s.count(anchor) == 1
WRAP = KERNEL + '''
static size_t iqk_fa_work_buffer_size_orig(const struct ggml_tensor * dst, int nth);

size_t iqk_fa_work_buffer_size(const struct ggml_tensor * dst, int nth) {
    size_t size = iqk_fa_work_buffer_size_orig(dst, nth);
#ifdef R5FA
    auto Qr = dst->src[0]; auto Kr = dst->src[1]; auto Vr = dst->src[2];
    if (r5fa::enabled() && Qr->type == GGML_TYPE_F32 && Kr->type == GGML_TYPE_F16 && Vr->type == GGML_TYPE_F16 &&
        Qr->ne[0] == r5fa::D && Vr->ne[0] == r5fa::D && Qr->ne[1] == 1 && Qr->ne[2] <= r5fa::kMaxHeads) {
        size = std::max(size, sizeof(float)*r5fa::slot_floats((int)Qr->ne[2])*nth);
    }
#endif
    return size;
}

static size_t iqk_fa_work_buffer_size_orig(const struct ggml_tensor * dst, int nth) {'''
s = s.replace(anchor, WRAP, 1)

# dispatch at the top of iqk_flash_attn_noalibi
old = '''    if (type_q != 0 || type_mask != 1 || max_bias > 0) return false;
'''
assert s.count(old) == 1
s = s.replace(old, old + '''
#ifdef R5FA
    if (r5fa::eligible(type_q, int_type_k_in, int_type_v, type_mask, Dk, Dv, neq1, neq2, neq3, nek2, nek3, nev2, nev3,
                       max_bias, softcap, mask, sinks, n_swa, indexer)) {
        return r5fa::run(neq2, nbq2, nek2, nbk2, nbv2, nb1, nek1, stride_k, stride_v, q, k, v, mask, scale, qkv,
                         work_buffer_in, barrier, barrier_data, ith, nth);
    }
#endif
''', 1)
fa.write_text(s)
print('patched')
