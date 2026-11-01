//
// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "iqk_config.h"
#include "iqk_mul_mat.h"
#include "iqk_flash_impl.h"
#include "ggml.h"

#if defined IQK_IMPLEMENT && defined GGML_IQK_FLASH_ATTENTION

#include <algorithm>
#include <cstdio>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <unordered_set>

namespace {
inline uint32_t simple_gcd(uint32_t a, uint32_t b) {
    while (a != b) {
        if (a > b) a -= b;
        else b -= a;
    }
    return a;
}
inline void accumulate_qkv(int Dv, float& M, float& S, float Mj, float Sj, float * Racc, const float * R) {
    if (Mj == -INFINITY) return;
    if (Mj > M) {
        if (M == -INFINITY) {
            std::memcpy(Racc, R, Dv*sizeof(float));
            S = Sj;
        } else {
            float c = exp(M - Mj);
            S = c*S + Sj;
            for (int i = 0; i < Dv; ++i) Racc[i] = c*Racc[i] + R[i];
        }
        M = Mj;
    } else {
        float c = exp(Mj - M);
        S += c*Sj;
        for (int i = 0; i < Dv; ++i) Racc[i] += c*R[i];
    }
}
}


// ---- round-5 decode flash attention (env IK_R5_FA=1) ------------------------------------
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#include "iqk_utils.h"
#include <immintrin.h>
#include <sys/mman.h>
#include <x86intrin.h>
#include <cstdlib>
namespace r5fa {
constexpr int D  = 256;
constexpr int JB = 32;
constexpr int kMaxHeads = 16;

static bool enabled() {
    static const bool e = [] { auto s = getenv("IK_R5_FA"); return s && atoi(s) > 0; }();
    return e;
}

static bool dbg_enabled() {
    static const bool e = [] { auto v = getenv("IK_R5_FA_DEBUG"); return v && atoi(v) > 0; }();
    return e;
}
static thread_local uint64_t r5ph[6];   // setup, scores, softmax, V accumulate, merge, calls
static inline uint64_t r5ph_now() { return dbg_enabled() ? __rdtsc() : 0; }

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
    uint64_t tp = r5ph_now();
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
    static const int pfb = [] { auto e = getenv("IK_R5_FA_PFB"); int v = e ? atoi(e) : 3; return v < 1 ? 1 : v; }();
    // upfront burst: the first pfb blocks (the first one is needed right away)
    for (int j = j0; j < std::min(j1, j0 + pfb*JB); ++j) {
        const char * kr = k + (long)j*stride_k;
        const char * vr = v + (long)j*stride_v;
        for (long o = 0; o < span_k; o += 64) _mm_prefetch(kr + o, _MM_HINT_T1);
        for (long o = 0; o < span_v; o += 64) _mm_prefetch(vr + o, _MM_HINT_T1);
    }
    if (dbg_enabled()) { uint64_t t = __rdtsc(); r5ph[0] += t - tp; tp = t; }
    for (int jb = j0; jb < j1; jb += JB) {
        const int nj = std::min(JB, j1 - jb);
        const int jn = jb + pfb*JB, nn = std::max(0, std::min(JB, j1 - jn));
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
        if (dbg_enabled()) { uint64_t t = __rdtsc(); r5ph[1] += t - tp; tp = t; }
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
        if (dbg_enabled()) { uint64_t t = __rdtsc(); r5ph[2] += t - tp; tp = t; }
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
        if (dbg_enabled()) { uint64_t t = __rdtsc(); r5ph[3] += t - tp; tp = t; }
    }
    for (int h = 0; h < nh; ++h) { m_out[h] = m[h]; l_out[h] = l[h]; }
}

// two-pass variant: K stream, exact softmax, V stream (see apply_r5fa2p.py)
static void partial2pass(int j0, int j1, int nkvh, int rk,
                    const char * q, long nbq2, const char * k, long stride_k, long nbk2,
                    const char * v, long stride_v, long nbv2, const uint16_t * mask, float scale,
                    float * m_out, float * l_out, float * O_all) {
    const int nh = nkvh*rk;
    const int n  = j1 - j0;
    std::memset(O_all, 0, sizeof(float)*nh*D);
    if (n <= 0) {
        for (int h = 0; h < nh; ++h) { m_out[h] = -INFINITY; l_out[h] = 0.0f; }
        return;
    }
    static const int pfb = [] { auto e = getenv("IK_R5_FA_PFB"); int x = e ? atoi(e) : 3; return x < 1 ? 1 : x; }();
    const int npad = (n + 7) & ~7;
    static thread_local std::vector<float> Sbuf;
    if ((int)Sbuf.size() < nh*npad + 8) Sbuf.resize(nh*npad + 8);
    float * S = Sbuf.data();
    alignas(32) float Qs[kMaxHeads*D];
    const __m256 vs = _mm256_set1_ps(scale);
    for (int h = 0; h < nh; ++h) {
        const float * qh = (const float *)(q + h*nbq2);
        for (int d = 0; d < D; d += 8) _mm256_store_ps(Qs + h*D + d, _mm256_mul_ps(vs, _mm256_loadu_ps(qh + d)));
    }
    const long span_k = (nkvh - 1)*nbk2 + 2*D;
    const long span_v = (nkvh - 1)*nbv2 + 2*D;
    const int pf = pfb*JB;   // rows ahead
    uint64_t tp = r5ph_now();

    // ---- pass A: scores, K stream only
    for (int j = 0; j < std::min(n, pf); ++j) {
        const char * kr = k + (long)(j0 + j)*stride_k;
        for (long o = 0; o < span_k; o += 64) _mm_prefetch(kr + o, _MM_HINT_T1);
    }
    for (int j = 0; j < n; j += 2) {
        for (int jj = j + pf; jj < j + pf + 2 && jj < n; ++jj) {
            const char * kr = k + (long)(j0 + jj)*stride_k;
            for (long o = 0; o < span_k; o += 64) _mm_prefetch(kr + o, _MM_HINT_T1);
        }
        for (int kh = 0; kh < nkvh; ++kh) {
            const char * kbase = k + kh*nbk2;
            const uint16_t * k0 = (const uint16_t *)(kbase + (long)(j0 + j)*stride_k);
            const uint16_t * k1 = j + 1 < n ? (const uint16_t *)(kbase + (long)(j0 + j + 1)*stride_k) : k0;
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
                S[(h0 + 0)*npad + j] = hsum(a00); S[(h0 + 1)*npad + j] = hsum(a10);
                S[(h0 + 2)*npad + j] = hsum(a20); S[(h0 + 3)*npad + j] = hsum(a30);
                if (j + 1 < n) {
                    S[(h0 + 0)*npad + j + 1] = hsum(a01); S[(h0 + 1)*npad + j + 1] = hsum(a11);
                    S[(h0 + 2)*npad + j + 1] = hsum(a21); S[(h0 + 3)*npad + j + 1] = hsum(a31);
                }
            }
        }
    }
    if (dbg_enabled()) { uint64_t t = __rdtsc(); r5ph[1] += t - tp; tp = t; }

    // ---- pass B: mask + exact softmax over the thread's range
    for (int h = 0; h < nh; ++h) {
        float * sh = S + h*npad;
        float M = -INFINITY;
        for (int j = 0; j < n; ++j) {
            const float x = sh[j] + _cvtsh_ss(mask[j0 + j]);
            sh[j] = x;
            M = std::max(M, x);
        }
        for (int j = n; j < npad; ++j) sh[j] = -INFINITY;
        m_out[h] = M;
        if (M == -INFINITY) { l_out[h] = 0.0f; std::memset(sh, 0, sizeof(float)*npad); continue; }
        const __m256 vm = _mm256_set1_ps(M);
        __m256 vsum = _mm256_setzero_ps();
        for (int j = 0; j < npad; j += 8) {
            const __m256 pj = v_expf(_mm256_sub_ps(_mm256_loadu_ps(sh + j), vm));
            _mm256_storeu_ps(sh + j, pj);
            vsum = _mm256_add_ps(vsum, pj);
        }
        l_out[h] = hsum(vsum);
    }
    if (dbg_enabled()) { uint64_t t = __rdtsc(); r5ph[2] += t - tp; tp = t; }

    // ---- pass C: V stream only, 32-row blocks (block stays in L1/L2 across the d-chunks)
    for (int j = 0; j < std::min(n, pf); ++j) {
        const char * vr = v + (long)(j0 + j)*stride_v;
        for (long o = 0; o < span_v; o += 64) _mm_prefetch(vr + o, _MM_HINT_T1);
    }
    for (int jb = 0; jb < n; jb += JB) {
        const int nj = std::min(JB, n - jb);
        for (int jj = jb + pf; jj < jb + pf + JB && jj < n; ++jj) {
            const char * vr = v + (long)(j0 + jj)*stride_v;
            for (long o = 0; o < span_v; o += 64) _mm_prefetch(vr + o, _MM_HINT_T1);
        }
        for (int kh = 0; kh < nkvh; ++kh) {
            const char * vbase = v + kh*nbv2;
            for (int h0 = kh*rk; h0 < (kh + 1)*rk; h0 += 4) {
                const float * p0 = S + (h0 + 0)*npad + jb, * p1 = S + (h0 + 1)*npad + jb;
                const float * p2 = S + (h0 + 2)*npad + jb, * p3 = S + (h0 + 3)*npad + jb;
                for (int d = 0; d < D; d += 16) {
                    float * o0 = O_all + (h0 + 0)*D + d, * o1 = O_all + (h0 + 1)*D + d, * o2 = O_all + (h0 + 2)*D + d, * o3 = O_all + (h0 + 3)*D + d;
                    __m256 c00 = _mm256_loadu_ps(o0), c01 = _mm256_loadu_ps(o0 + 8), c10 = _mm256_loadu_ps(o1), c11 = _mm256_loadu_ps(o1 + 8);
                    __m256 c20 = _mm256_loadu_ps(o2), c21 = _mm256_loadu_ps(o2 + 8), c30 = _mm256_loadu_ps(o3), c31 = _mm256_loadu_ps(o3 + 8);
                    for (int j = 0; j < nj; ++j) {
                        const uint16_t * vr = (const uint16_t *)(vbase + (long)(j0 + jb + j)*stride_v) + d;
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
    if (dbg_enabled()) { uint64_t t = __rdtsc(); r5ph[3] += t - tp; }
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
    static const bool streamtest = [] { auto e = getenv("IK_R5_FA_STREAMTEST"); return e && atoi(e) > 0; }();
    if (streamtest && j1 > j0) {
        static thread_local char * ring = nullptr;
        static thread_local size_t ring_off = 0;
        static thread_local uint64_t tk, tr, bytes, nc;
        constexpr size_t kRing = (size_t)64 << 20;
        if (!ring) { ring = (char *)aligned_alloc(2u << 20, kRing); madvise(ring, kRing, MADV_HUGEPAGE); memset(ring, 1, kRing); }
        const long span = (long)(nek2 - 1)*nbk2 + 2*D;
        __m256 acc = _mm256_setzero_ps();
        const uint64_t a = __rdtsc();
        for (int j = j0; j < j1; ++j) {
            const char * kr = (const char *)k + (long)j*stride_k;
            const char * vr = (const char *)v + (long)j*stride_v;
            for (long o = 0; o < span; o += 32) {
                acc = _mm256_add_ps(acc, _mm256_loadu_ps((const float *)(kr + o)));
                acc = _mm256_add_ps(acc, _mm256_loadu_ps((const float *)(vr + o)));
            }
        }
        const uint64_t b = __rdtsc();
        const size_t n = (size_t)(j1 - j0)*span*2;
        if (ring_off + n > kRing/2) ring_off = 0;
        for (size_t o = 0; o < n/2; o += 32) {
            acc = _mm256_add_ps(acc, _mm256_load_ps((const float *)(ring + ring_off + o)));
            acc = _mm256_add_ps(acc, _mm256_load_ps((const float *)(ring + kRing/2 + ring_off + o)));
        }
        const uint64_t c = __rdtsc();
        for (size_t o = n/2; o < n + n/2; o += 32) acc = _mm256_add_ps(acc, _mm256_load_ps((const float *)(ring + ring_off + o)));
        ring_off += (n + n/2 + 4095) & ~(size_t)4095;
        const uint64_t d = __rdtsc();
        const int jj0 = j1 < n_eff ? j1 : 0, jj1 = std::min(n_eff, jj0 + (j1 - j0));
        for (int j = jj0; j < jj1; ++j) { const char * kr = (const char *)k + (long)j*stride_k;
            for (long o = 0; o < span; o += 32) acc = _mm256_add_ps(acc, _mm256_loadu_ps((const float *)(kr + o))); }
        for (int j = jj0; j < jj1; ++j) { const char * vr = (const char *)v + (long)j*stride_v;
            for (long o = 0; o < span; o += 32) acc = _mm256_add_ps(acc, _mm256_loadu_ps((const float *)(vr + o))); }
        const uint64_t e = __rdtsc();
        volatile float sink = _mm256_cvtss_f32(acc); (void)sink;
        static thread_local uint64_t t_kvi, t_r2, t_r1, t_kvs, b_kvi, b_kvs;
        t_kvi += b - a; t_r2 += c - b; t_r1 += d - c; t_kvs += e - d; b_kvi += n; b_kvs += (size_t)(jj1 - jj0)*span*2;
        if (++nc % 640 == 0) {
            fprintf(stderr, "r5fa-stream2 ith %d: KVi %.2f KVs %.2f R2 %.2f R1 %.2f bytes/tick\n", ith,
                    (double)b_kvi/t_kvi, (double)b_kvs/t_kvs, (double)b_kvi/t_r2, (double)b_kvi/t_r1);
            t_kvi = t_r2 = t_r1 = t_kvs = b_kvi = b_kvs = 0;
        }
    }
    static const bool two_pass = [] { auto e = getenv("IK_R5_FA_2PASS"); return e && atoi(e) > 0; }();
    const uint64_t t0 = dbg ? __rdtsc() : 0;
    (two_pass ? partial2pass : partial)(j0, j1, nek2, rk, (const char *)q, nbq2, (const char *)k, stride_k, nbk2,
            (const char *)v, stride_v, nbv2, mk, scale, mine, mine + neq2, mine + 2*neq2);
    const uint64_t t1 = dbg ? __rdtsc() : 0;
    static const bool twice = [] { auto e = getenv("IK_R5_FA_TWICE"); return e && atoi(e) > 0; }();
    if (twice) {
        static thread_local uint64_t p1, p2, nc;
        const uint64_t a = __rdtsc();
        partial(j0, j1, nek2, rk, (const char *)q, nbq2, (const char *)k, stride_k, nbk2, (const char *)v, stride_v, nbv2,
                mk, scale, mine, mine + neq2, mine + 2*neq2);
        p2 += __rdtsc() - a; p1 += t1 - t0;
        if (++nc % 640 == 0) {
            fprintf(stderr, "r5fa-twice ith %d positions %d: pass1 %.0f  pass2 %.0f ticks/call\n", ith, j1 - j0, p1/640.0, p2/640.0);
            p1 = p2 = 0;
        }
    }
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
    const uint64_t tm0 = r5ph_now();
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
    if (dbg_enabled()) {
        r5ph[4] += __rdtsc() - tm0;
        if (++r5ph[5] % 640 == 0) {
            const double n = 640.0;
            fprintf(stderr, "r5fa-phase ith %d (positions %d): setup %.0f  scores %.0f  softmax %.0f  V %.0f  merge %.0f ticks/call\n",
                    ith, j1 - j0, r5ph[0]/n, r5ph[1]/n, r5ph[2]/n, r5ph[3]/n, r5ph[4]/n);
            for (int i = 0; i < 5; ++i) r5ph[i] = 0;
        }
    }
    return true;
}
} // namespace r5fa
#define R5FA 1
#endif
// ---- end round-5 decode flash attention -------------------------------------------------

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

static size_t iqk_fa_work_buffer_size_orig(const struct ggml_tensor * dst, int nth) {
    auto Q = dst->src[0];
    auto K = dst->src[1];
    auto V = dst->src[2];
    auto indexer = dst->src[5];
    if (indexer && indexer->type == GGML_TYPE_I32 && indexer->ne[0] < K->ne[1] &&
        Q->ne[3] == 1 && K->ne[3] == 1 && V->ne[3] == 1 && (K->ne[2] == 1 || Q->ne[1] > 1)) {
        auto row_size_k = ggml_row_size(K->type, K->ne[0])*K->ne[2];
        auto row_size_v = ggml_row_size(V->type, V->ne[0])*K->ne[2];
        auto work_size  = (row_size_k + row_size_v + 64) * indexer->ne[0];
        size_t result = work_size * nth;
        if (Q->ne[1] == 1) result += 512*sizeof(float);
        return result;
        //return work_size * nth;
    }
    int rk2 = Q->ne[2]/K->ne[2];
    size_t size = 0;
    if (Q->ne[1] >= 8 && K->type == GGML_TYPE_Q8_0) {
        size = ggml_row_size(GGML_TYPE_Q8_0, K->ne[0]) * K->ne[1]*K->ne[2]*K->ne[3];
    }
    if (Q->ne[1] == 1 && Q->ne[3] == 1 && Q->ne[2]/K->ne[2] > 1 && nth >= 1 && K->ne[1]/32 > 1) {
        if (K->ne[2] > 1) {
            // Cover any nek1 that iqk_flash_attn_noalibi narrows to at run time.
            int nstep_k = K->ne[2]*((K->ne[1] + 31)/32);
            if (nstep_k > 4*nth + K->ne[2]) nstep_k = 4*nth + K->ne[2];
            size_t result_size = (V->ne[0] + 16)*Q->ne[2]/K->ne[2]*sizeof(float);
            size += nstep_k*result_size;
            return size;
        }
        // With nth >= rk2 the consumer can fall through to the chunked loop, which uses up to 4*nth slots.
        int nstep_k = nth;
        if (nth >= rk2) {
            nstep_k = (K->ne[1] + 31)/32;
            if (nstep_k > 4*nth) nstep_k = 4*nth;
            if (nstep_k < nth) nstep_k = nth;
        }
        size_t result_size = (V->ne[0] + 16)*rk2*sizeof(float);
        size += nstep_k*result_size;
        return size;
    }
    return size;
}

static inline const std::unordered_set<ggml_type> & supported_kv_types() {
#ifdef GGML_IQK_FA_ALL_QUANTS
    static std::unordered_set<ggml_type> k_supported = {
        GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q8_KV, GGML_TYPE_Q6_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL
    };
#else
    static std::unordered_set<ggml_type> k_supported = {
        GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q8_KV, GGML_TYPE_Q6_0,
    };
#endif
    return k_supported;
}

static inline bool are_kv_types_supported(ggml_type type_k, ggml_type type_v) {
    if (type_k == GGML_TYPE_BF16) {
        if (type_v != type_k) {
            return false;
        }
#ifdef __AVX512BF16__
        return true;
#else
        return false;
#endif
    }
    auto & supported = supported_kv_types();
    auto it_k = supported.find(type_k);
    auto it_v = supported.find(type_v);
    return it_k != supported.end() && it_v != supported.end();
}

// TODO: get the ggml_type enum here without polution
//
extern "C" IQK_API bool iqk_flash_attn_noalibi(int type_q, int type_mask, float max_bias,
                            int neq3, int neq2, long nbq3, long nbq2,
                            int nek3, int nek2, long nbk3, long nbk2,
                            int nev3, int nev2, long nbv3, long nbv2,
                            int ne2,  int ne1,  long nb1,
                            int int_type_k_in,      // type of k
                            int int_type_v,         // type of v
                            int Dk,                 // K head size
                            int Dv,                 // V head size
                            int neq1,               // number of columns in q
                            int nek1,               // number of rows in k
                            int stride_q,           // distance between q columns in bytes
                            int stride_k,           // distance between k rows in bytes
                            int stride_v,           // distance between v rows in bytes
                            int stride_m,           // distance between mask rows (in bytes
                            const void  * q,        // q matrix.
                            const void  * k,        // k matrix. Assumed to be fp16, nq x nk elements
                            const void  * v,        // v matrix. Assumed to be fp16, nq x nk elements
                            const void  * mask,     // mask. If not null, assumed to be fp16. nq x nk elements
                            const void  * sinks,    // mask. If not null, assumed to be fp16. nq x nk elements
                            float         scale,    // scale applied before softmax
                            float         softcap,  // if > 0, a "soft-cap" operation is applied before softmax
                            float       * qkv,      // v*softmax(scale*(k*q))
                            [[maybe_unused]] void * work_buffer_in, [[maybe_unused]] barrier_t barrier, [[maybe_unused]] void * barrier_data,
                            int ith, int nth, int n_swa, [[maybe_unused]] ggml_tensor * indexer) {

    if (type_q != 0 || type_mask != 1 || max_bias > 0) return false;

#ifdef R5FA
    if (r5fa::eligible(type_q, int_type_k_in, int_type_v, type_mask, Dk, Dv, neq1, neq2, neq3, nek2, nek3, nev2, nev3,
                       max_bias, softcap, mask, sinks, n_swa, indexer)) {
        return r5fa::run(neq2, nbq2, nek2, nbk2, nbv2, nb1, nek1, stride_k, stride_v, q, k, v, mask, scale, qkv,
                         work_buffer_in, barrier, barrier_data, ith, nth);
    }
#endif

    if (indexer && indexer->type == GGML_TYPE_I32) {
        if (indexer->ne[0] < nek1 && neq3 == 1 && nek3 == 1 && nev3 == 1 && (nek2 == 1 || neq1 > 1)) {
            // Workbuffer: we need
            // * indexer->ne[0] * sizeof(ggml_half) to extract the mask for a row
            // * indexer->ne[0] * ggml_row_size(int_type_k_in, Dk) to extract the selected K cache entries
            // * indexer->ne[0] * ggml_row_size(int_type_v, Dv) to extract the selected V cache entries
            auto row_size_k = ggml_row_size(ggml_type(int_type_k_in), Dk)*nek2;
            auto row_size_v = ggml_row_size(ggml_type(int_type_v   ), Dv)*nek2;
            auto work_size  = (row_size_k + row_size_v + 64) * indexer->ne[0];
            ggml_fp16_t h_inf = ggml_fp32_to_fp16(-INFINITY);
            ggml_fp16_t h_zero = ggml_fp32_to_fp16(0.0f);
            int  nkv = indexer->ne[0];
            if (neq1 == 1) {
                GGML_ASSERT(neq2 <= 256);
                int npt = (neq2 + nth - 1)/nth;
                int ith_mid = nth;
                int neq2_this_thread = npt;
                int first = ith*npt;
                if (npt*nth > neq2) {
                    ith_mid = neq2 - nth*(npt - 1);
                    if (ith >= ith_mid) {
                        --neq2_this_thread;
                        //if (neq2_this_thread < 1) return true;
                        first = ith_mid*npt + (ith - ith_mid)*neq2_this_thread;
                    }
                }
                auto idx = (const int *)indexer->data;
                auto M = (const ggml_fp16_t *)mask;
                auto work_k = (char *)work_buffer_in;
                auto work_v = work_k + row_size_k*indexer->ne[0];
                auto work_m = (ggml_fp16_t *)(work_v + row_size_v*indexer->ne[0]) + indexer->ne[0]*ith;
                int last_found = -1;
                for (int j = 0; j < nkv; ++j) {
                    if (idx[j] >= 0) {
                        last_found = j;
                        work_m[j] = M ? M[idx[j]] : h_zero;
                        if (j % nth == ith) {
                            std::memcpy(work_k + row_size_k*j, ((const char *)k + idx[j]*stride_k), row_size_k);
                            if (k != v) {
                                std::memcpy(work_v + row_size_v*j, ((const char *)v + idx[j]*stride_v), row_size_v);
                            }
                        }
                    } else {
                        work_m[j] = h_inf;
                        if (j % nth == ith) {
                            std::memset(work_k + row_size_k*j, 0, row_size_k);
                            if (k != v) {
                                std::memset(work_v + row_size_v*j, 0, row_size_v);
                            }
                        }
                    }
                }
                barrier(barrier_data);
                if (last_found < 0 || neq2_this_thread < 1) return true;
                ++last_found;
                int this_nkv = 32*((last_found + 31)/32);
                auto this_q = (const char *)q + first*nbq2;
                auto this_qkv = qkv + first*nb1/sizeof(float);
                if (!iqk_flash_attn_impl(int_type_k_in, int_type_v,
                         Dk, Dv, neq2_this_thread, this_nkv, nbq2, row_size_k, row_size_v, 0, Dv,
                         (const float *)this_q, work_k, k == v ? work_k : work_v, work_m, sinks ? (const float *)sinks + first : nullptr, 1,
                         scale, softcap,
                         this_qkv, nullptr, nullptr)) return false;
                return true;
            }
            int npt = (neq1 + nth - 1)/nth;
            int ith_mid = nth;
            int neq1_this_thread = npt;
            int first = ith*npt;
            if (npt*nth > neq1) {
                ith_mid = neq1 - nth*(npt - 1);
                if (ith >= ith_mid) {
                    --neq1_this_thread;
                    if (neq1_this_thread < 1) return true;
                    first = ith_mid*npt + (ith - ith_mid)*neq1_this_thread;
                }
            }
            auto work_k = (char *)work_buffer_in + ith*work_size;
            auto work_v = work_k + row_size_k*indexer->ne[0];
            auto work_m = (ggml_fp16_t *)(work_v + row_size_v*indexer->ne[0]);
            for (int iq = first; iq < first + neq1_this_thread; ++iq) {
                auto idx = (const int *)((const char *)indexer->data + iq*indexer->nb[1]);
                auto M = mask ? (const ggml_fp16_t *)((const char *)mask + iq*stride_m) : nullptr;
                int last_found = -1;
                for (int j = 0; j < nkv; ++j) {
                    if (idx[j] >= 0) {
                        std::memcpy(work_k + row_size_k*j, ((const char *)k + idx[j]*stride_k), row_size_k);
                        if (k != v) {
                            std::memcpy(work_v + row_size_v*j, ((const char *)v + idx[j]*stride_v), row_size_v);
                        }
                        work_m[j] = M ? M[idx[j]] : h_zero;
                        last_found = j;
                    } else {
                        std::memset(work_k + row_size_k*j, 0, row_size_k);
                        if (k != v) {
                            std::memset(work_v + row_size_v*j, 0, row_size_v);
                        }
                        work_m[j] = h_inf;
                    }
                }
                if (last_found < 0) continue;
                ++last_found;
                int this_nkv = 32*((last_found + 31)/32);
                auto this_q = (const char *)q + iq*stride_q;
                auto this_qkv = qkv + iq*ne1*nb1/sizeof(float);
                for (int ik2 = 0; ik2 < nek2; ++ik2) {
                    if (!iqk_flash_attn_impl(int_type_k_in, int_type_v,
                                Dk, Dv, neq2/nek2, this_nkv, nbq2, row_size_k, row_size_v, 0, Dv,
                                (const float *)(this_q + ik2*(neq2/nek2)*nbq2),
                                work_k + ik2*row_size_k/nek2,
                                k == v ? work_k + ik2*row_size_k/nek2: work_v + ik2*row_size_v/nek2,
                                work_m, (const float *)sinks, 1, scale, softcap,
                                this_qkv + ik2*(neq2/nek2)*nb1/sizeof(float), nullptr, nullptr)) return false;
                }
            }
            return true;
        }
    }

    if (auto type_k = ggml_type(int_type_k_in), type_v = ggml_type(int_type_v); !are_kv_types_supported(type_k, type_v)) {
        if (ith == 0) {
            fprintf(stderr, "\n==================== K cache %s coupled with V cache %s is not a supported combination on the CPU backend.\n",
                    ggml_type_name(type_k), ggml_type_name(type_v));
            auto & supported = supported_kv_types();
            fprintf(stderr, "Supported types are:\n");
            for (auto type : supported) {
                fprintf(stderr, "    %s\n", ggml_type_name(type));
            }
            fprintf(stderr, "    Warning: ik_llama.cpp does not support Q5_0 or Q5_1 KV cache on the CPU.\n");
#ifdef __AVX512BF16__
            fprintf(stderr, "    %s, but only if K and V are both %s\n", ggml_type_name(GGML_TYPE_BF16), ggml_type_name(GGML_TYPE_BF16));
#endif
#ifndef GGML_IQK_FA_ALL_QUANTS
            fprintf(stderr, "    To enable q4_0, q4_1, and iq4_nl KV cache types, recompile with -DGGML_IQK_FA_ALL_QUANTS=ON\n");
#endif
        }
        barrier(barrier_data);
        GGML_ABORT("Fatal error");
    }

    if (n_swa > 0 && mask) {
        constexpr int kMinBatch = 256;
        int ntokens = std::max(kMinBatch, neq1);
        int nblock  = (ntokens + n_swa + kMinBatch - 1)/kMinBatch;
        int first   = nek1 - nblock*kMinBatch;
        if (first > 0) {
            k = (const char *)k + int64_t(first)*stride_k;
            v = (const char *)v + int64_t(first)*stride_v;
            mask = (const uint16_t *)mask + first;
            nek1 -= first;
        }
    }

    int rk2 = neq2/nek2;
    int rv2 = neq2/nev2;
    int rk3 = neq3/nek3;
    int rv3 = neq3/nev3;

    int first_k = 0, last_k = nek1;
    if (neq3 == 1 && rk2 > 1 && neq1 == 1 && nek1 > 256 && mask) {
        // This is a quick hack for SWA models.
        // Given that the mask is the same for all layers, ideally we should determine the
        // cache bounds once, and reuse for the whole graph. But even with this simple hack
        // we get non-negligible performance gains for SWA models and long context.
        auto umask = (const uint16_t *)mask;
        for (; first_k < last_k; ++first_k) {
            if (umask[first_k] == 0) break;
        }
        if (first_k == last_k) {
            fprintf(stderr, "============================== %s: found empty attention mask: nek1 = %d, first_k = %d\n", __func__, nek1, first_k);
            GGML_ABORT("Fatal error");
        }
        for (; last_k > first_k; --last_k) {
            if (umask[last_k-1] == 0) break;
        }
        int non = 32*((last_k - first_k + 31)/32);
        first_k = std::max(0, last_k - non);
        last_k = std::min(first_k + non, nek1);
        //printf("nek1 = %d, first = %d, last = %d\n", nek1, first, last);
        if (last_k - first_k <= 3*nek1/4 && (last_k - first_k)%32 == 0) {
            //printf("Reducing from %d to %d\n", nek1, last_k - first_k);
            k = (const void *)((const char *)k + first_k*stride_k);
            v = (const void *)((const char *)v + first_k*stride_v);
            mask = (const void *)((const uint16_t *)mask + first_k);
            nek1 = last_k - first_k;
        }
    }

    int int_type_k = int_type_k_in;
    auto work_buffer = work_buffer_in;
    if (neq1 >= 8) {
        uint64_t row_size = 0;
        work_buffer = iqk_repack_k(int_type_k, Dk, nek1, nek2, nek3, stride_k, nbk2, nbk3, k, work_buffer_in, ith, nth, int_type_k, row_size);
        if (int_type_k != int_type_k_in) {
            stride_k = row_size;
            nbk2 = stride_k*nek1;
            nbk3 = nbk2*nek2;
            k = work_buffer_in;
            barrier(barrier_data);
        }
    }
    //uint64_t row_size = 0;
    //auto work_buffer = iqk_repack_k(int_type_k, Dk, nek1, nek2, nek3, stride_k, nbk2, nbk3, k, work_buffer_in, ith, nth, int_type_k, row_size);
    //if (int_type_k != int_type_k_in) {
    //    stride_k = row_size;
    //    nbk2 = stride_k*nek1;
    //    nbk3 = nbk2*nek2;
    //    k = work_buffer_in;
    //    barrier(barrier_data);
    //}

    // Getting confused all the time about where to load data from and store the results to
    // (especially when combining the results from the threads).
    // So, for now, making it work just for MLA (nek2 = 1).
    // I think it would also speed up things for GQA, but I'm leaving this for another day.
    if (neq3 == 1 && rk2 > 1 && neq1 == 1 && nth >= 1 && nek1/32 > 1 && nek2 == 1) {
        int nstep_k = nek1/32;
        if (nstep_k >= 4*nth) {
            int nstep_k_per_thread = (nstep_k + nth - 1)/nth;
            int ith_mid = nth;
            int nstep_k_this_thread = nstep_k_per_thread;
            if (nstep_k_per_thread*nth > nstep_k) {
                ith_mid = nstep_k - nth*(nstep_k_per_thread - 1);
                if (ith >= ith_mid) --nstep_k_this_thread;
            }
            //if (ith == 0) fprintf(stderr, "nstep_k = %d, nstep_k_per_thread = %d, ith_mid = %d\n", nstep_k, nstep_k_per_thread, ith_mid);
            nstep_k_per_thread *= 32;
            nstep_k_this_thread *= 32;

            auto kv_offset = ith <= ith_mid ? ith*nstep_k_per_thread
                                           : ith_mid*nstep_k_per_thread + (ith - ith_mid)*nstep_k_this_thread;
            auto kth = (const char *)k + kv_offset*stride_k;
            auto vth = (const char *)v + kv_offset*stride_v;
            auto qth = (const char *)q;
            auto mth = mask ? (const char *)mask + kv_offset*sizeof(uint16_t) : nullptr; // we don't have ggml_half available here

            auto work = (char *)work_buffer;
            auto size_thread = (Dv + 16)*rk2*sizeof(float);
            auto result_buffer = work;
            auto work_this_thread = (float *)(result_buffer + ith*size_thread);
            if (!iqk_flash_attn_impl(int_type_k, int_type_v,
                     Dk, Dv, rk2, nstep_k_this_thread, nbq2, stride_k, stride_v, 0, Dv, //Dk*sizeof(uint16_t), Dv,
                     (const float *)qth, (const void *)kth, (const void *)vth, (const void *)mth, nullptr, 0,
                     scale, softcap,
                     work_this_thread, work_this_thread + (Dv+0)*rk2, work_this_thread + (Dv+1)*rk2)) return false;

            barrier(barrier_data);

            for (int j = ith; j < rk2; j += nth) {
                auto Racc = qkv + j*nb1/sizeof(float);
                float M = -INFINITY, S = 0;
                for (int jth = 0; jth < nth; ++jth) {
                    auto R = (const float *)(result_buffer + jth*size_thread);
                    auto Mj = R + Dv*rk2;
                    auto Sj = Mj + rk2;
                    R += j*Dv;
                    accumulate_qkv(Dv, M, S, Mj[j], Sj[j], Racc, R);
                }
                float norm = S > 0 ? 1/S : 1;
                for (int i = 0; i < Dv; ++i) Racc[i] *= norm;
            }
            return true;
        }
        int gcd_k   = simple_gcd(nstep_k, nth);
        if (gcd_k >= 1) {
            int nth_k = nth/gcd_k;
            int ith_k = ith%gcd_k;
            int ith_q = ith/gcd_k;
            int nq_per_thread = (rk2 + nth_k - 1)/nth_k;
            if (nq_per_thread > 1) {
                int ith_mid = nth_k;
                int nq_this_thread = nq_per_thread;
                if (nq_per_thread*nth_k > rk2) {
                    ith_mid = rk2 - nth_k*(nq_per_thread - 1);
                    if (ith_q >= ith_mid) --nq_this_thread;
                }
                int j_mid = ith_mid*nq_per_thread;
                auto work = (char *)work_buffer;
                auto size_thread = (Dv + 16)*nq_per_thread*sizeof(float);
                auto result_buffer = work;

                auto kth = (const char *)k + ith_k*(nek1/gcd_k)*stride_k;
                auto vth = (const char *)v + ith_k*(nek1/gcd_k)*stride_v;
                auto q_offset = ith_q < ith_mid ? ith_q*nq_per_thread*nbq2 : (ith_mid*nq_per_thread + (ith_q - ith_mid)*nq_this_thread)*nbq2;
                auto qth = (const char *)q + q_offset;
                auto mth = mask ? (const char *)mask + ith_k*(nek1/gcd_k)*sizeof(uint16_t) : nullptr; // we don't have ggml_half available here

                // Each thread will produce a result of size Dv*nq_this_thread*sizeof(float)
                // In addition, we need M, S for the nq_this_thread rows the thread is processing
                // => (Dv + 2)*nq_per_thread*sizeof(float). We use (Dv + 16) instead to make sure threads are not
                // writing onto the same cache line.
                auto work_this_thread = (float *)(result_buffer + ith*size_thread);
                if (!iqk_flash_attn_impl(int_type_k, int_type_v,
                            Dk, Dv, nq_this_thread, nek1/gcd_k, nbq2, stride_k, stride_v, 0, Dv, //Dk*sizeof(uint16_t), Dv,
                            (const float *)qth, (const void *)kth, (const void *)vth, (const void *)mth, nullptr, 0,
                            scale, softcap,
                            work_this_thread, work_this_thread + (Dv+0)*nq_this_thread, work_this_thread + (Dv+1)*nq_this_thread)) return false;

                barrier(barrier_data);

                // There are nek1/gcd_k contributions for each j that we need to sum up
                // Thread i computed k/v (i%gcd_k)*(nek1/gcd_k) for j (i/gcd_k)*(rk2/nth_k)...((i/gcd_k)+1)*(rk2/nth_k) and results at offset i*size_thread

                // TODO: simdify this
                // TODO: if nth > rk2, have threads process portions of the rows instead of entire rows as it is now
                for (int j = ith; j < rk2; j += nth) {
                    auto Racc = qkv + j*nb1/sizeof(float);
                    float M = -INFINITY, S = 0;
                    int jth_first, jj, nq_this_j;
                    if (j < j_mid) {
                        jth_first = j/nq_per_thread;
                        jj = j%nq_per_thread;
                        nq_this_j = nq_per_thread;
                    } else {
                        jth_first = ith_mid + (j - j_mid)/(nq_per_thread-1);
                        jj = (j - j_mid)%(nq_per_thread-1);
                        nq_this_j = nq_per_thread - 1;
                    }
                    jth_first *= gcd_k;
                    for (int jth = jth_first; jth < jth_first + gcd_k; ++jth) {
                        auto R = (const float *)(result_buffer + jth*size_thread);
                        auto Mj = R + Dv*nq_this_j;
                        auto Sj = Mj + nq_this_j;
                        R += jj*Dv;
                        accumulate_qkv(Dv, M, S, Mj[jj], Sj[jj], Racc, R);
                    }
                    float norm = S > 0 ? 1/S : 1;
                    for (int i = 0; i < Dv; ++i) Racc[i] *= norm;
                }
                return true;
            }
        }
    }

    if (neq3 == 1 && rk2 > 1 && rk2 == rv2 && neq1 == 1 && nth >= 1 && nek2*nek1 >= 32*nth) {
        auto result_size = (Dv + 16)*rk2*sizeof(float);
        int gcd = simple_gcd(nek2, nth);
        int nth_k  = nth/gcd;
        int nek2_k = nek2/gcd;
        int nchunk = nek2_k*nek1/32;
        int npt = (nchunk + nth_k - 1)/nth_k;
        int nk;
        if (npt*nth_k == nchunk) {
            nk = 32 * (nek2*nek1/(32*nth));
        } else {
            //int nm = std::max(1, npt/8);
            int nm = 1;
            while (true) {
                if (nm*4 >= npt) break;
                nm *= 2;
            }
            nk = 32*nm;
        }
        //int nk = 32 * (nek2*nek1/(32*nth));
        int nkk = (nek1 + nk - 1)/nk;
        int nstep_k = nek2*nkk;
        //if (ith == 0) {
        //    static int last_nek1 = 0;
        //    if (last_nek1 != nek1) {
        //        printf("rk2 = %d, nek1 = %d, nek2 = %d, nk = %d, nkk = %d, nstep_k = %d\n", (int)rk2, (int)nek1, (int)nek2, nk, nkk, nstep_k);
        //        last_nek1 = nek1;
        //    }
        //}
        for (int istep_k = ith; istep_k < nstep_k; istep_k += nth) {
            int ik02 = istep_k/nkk;
            int ik01 = nk*(istep_k - ik02*nkk);
            int this_nk = ik01 + nk <= nek1 ? nk : nek1 - ik01;
            if (this_nk <= 0) break;
            auto this_result = (float *)((char *)work_buffer + istep_k*result_size);
            auto this_q = (const float *)((const char *)q + ik02*rk2*nbq2);
            auto this_k = (const char *)k + ik01*stride_k + ik02*nbk2;
            auto this_v = (const char *)v + ik01*stride_v + ik02*nbv2;
            auto this_m = mask ? (const char *)mask + ik01*sizeof(uint16_t) : nullptr; // we don't have ggml_half available here
            if (!iqk_flash_attn_impl(int_type_k, int_type_v,
                     Dk, Dv, rk2, this_nk, nbq2, stride_k, stride_v, 0, Dv,
                     this_q, (const void *)this_k, (const void *)this_v, (const void *)this_m, nullptr, 0,
                     scale, softcap, this_result, this_result + (Dv+0)*rk2, this_result + (Dv+1)*rk2)) return false;
        }

        barrier(barrier_data);

        // We have nkk results for each head
        for (int iq2 = ith; iq2 < neq2; iq2 += nth) {
            // ik02*rk2 + il = iq2 (il = 0...rk2-1) => ik02 = iq2/rk2, il = iq2%rk2;
            int ik02 = iq2/rk2;
            int il = iq2 - ik02*rk2;
            auto Racc = qkv + iq2*nb1/sizeof(float);
            //std::memset(Racc, 0, Dv*sizeof(float));
            float M = -INFINITY, S = 0;
            for (int ikk = 0; ikk < nkk; ++ikk) {
                int istep_k = ik02*nkk + ikk;
                auto this_result = (float *)((char *)work_buffer + istep_k*result_size);
                const float * R  = this_result + il*Dv;
                const float * Mj = this_result + Dv*rk2;
                const float * Sj = Mj + rk2;
                accumulate_qkv(Dv, M, S, Mj[il], Sj[il], Racc, R);
            }
            if (sinks) {
                float s = ((const float *)sinks)[iq2];
                if (s > M) {
                    float m = expf(M - s);
                    for (int i = 0; i < Dv; ++i) Racc[i] *= m;
                    S = S*m + 1;
                } else {
                    S += expf(s - M);
                }
            }
            float norm = S > 0 ? 1/S : 1;
            for (int i = 0; i < Dv; ++i) Racc[i] *= norm;
        }
        return true;
    }

    // I keep changing my mind what is the best strategy to split the threads when processing
    // multiple heads. This is my current thinking, the commented out code below was the previous.
    int ntg = nth/simple_gcd(neq2*neq3, nth);
    int neq1g = (neq1 + ntg - 1)/ntg;
    //int64_t work_per_slice = D*nek1*neq1;
    //int ntg = 1;
    //
    // When neq1 is large, it is better to have more than one thread process one (iq2,iq3) matrix
    // But we also want each thread to process the same amount of rows, so neq1 must be a multiple of
    // the number of threads processing the (iq2, iq3) matrix.
    //
    //if (neq1 >= 8*nth) {
    //    if      (nth%8 == 0 && neq1%8 == 0 && work_per_slice >= (1 << 23)) ntg = 8;
    //    else if (nth%4 == 0 && neq1%4 == 0 && work_per_slice >= (1 << 21)) ntg = 4;
    //    else if (nth%2 == 0 && neq1%2 == 0 && work_per_slice >= (1 << 19)) ntg = 2;
    //}
    int counter = 0;
    for (int64_t iq3 = 0; iq3 < neq3; iq3++) {
        for (int64_t iq2 = 0; iq2 < neq2; iq2++) {
            auto sinksf = sinks ? (const float *)sinks + iq2 : nullptr;
            if (counter++ % (nth/ntg) == ith/ntg) {
                int iq1 = (ith%ntg)*neq1g;
                int this_neq1 = std::min(neq1g, neq1-iq1);
                if (this_neq1 > 0) {
                if (!iqk_flash_attn_impl(int_type_k, int_type_v,
                        Dk, Dv, this_neq1, nek1, stride_q, stride_k, stride_v, stride_m, ne1*nb1/sizeof(float),
                        (const float *)((const char *)q + iq2*nbq2 + iq3*nbq3 + iq1*stride_q),
                        (const void  *)((const char *)k + iq2/rk2*nbk2 + iq3/rk3*nbk3),
                        (const void  *)((const char *)v + iq2/rv2*nbv2 + iq3/rv3*nbv3),
                        mask ? (const void  *)((const char *)mask + iq1*stride_m) : nullptr, sinksf, 0,
                        scale, softcap,
                        (float *)((char *)qkv + (iq3*ne2*ne1 + iq2 + iq1*ne1)*nb1), nullptr, nullptr)) return false;
                }
            }
        }
    }

    return true;
}

#else

bool iqk_flash_attn_noalibi([[maybe_unused]] int type_q, [[maybe_unused]] int type_mask, [[maybe_unused]] float max_bias,
                            [[maybe_unused]] int neq3, [[maybe_unused]] int neq2, [[maybe_unused]] long nbq3, [[maybe_unused]] long nbq2,
                            [[maybe_unused]] int nek3, [[maybe_unused]] int nek2, [[maybe_unused]] long nbk3, [[maybe_unused]] long nbk2,
                            [[maybe_unused]] int nev3, [[maybe_unused]] int nev2, [[maybe_unused]] long nbv3, [[maybe_unused]] long nbv2,
                            [[maybe_unused]] int ne2,  [[maybe_unused]] int ne1,  [[maybe_unused]] long nb1,
                            [[maybe_unused]] int type_k,             // type of k
                            [[maybe_unused]] int type_v,             // type of v
                            [[maybe_unused]] int Dk,                 // K head size
                            [[maybe_unused]] int Dv,                 // V head size
                            [[maybe_unused]] int nq,                 // number of columns in q
                            [[maybe_unused]] int nk,                 // number of rows in k
                            [[maybe_unused]] int stride_q,           // distance between q columns in bytes
                            [[maybe_unused]] int stride_k,           // distance between k rows in bytes
                            [[maybe_unused]] int stride_v,           // distance between v rows in bytes
                            [[maybe_unused]] int stride_m,           // distance between mask rows (in bytes
                            [[maybe_unused]] const void  * q,        // q matrix.
                            [[maybe_unused]] const void  * k,        // k matrix. Assumed to be fp16, nq x nk elements
                            [[maybe_unused]] const void  * v,        // v matrix. Assumed to be fp16, nq x nk elements
                            [[maybe_unused]] const void  * mask,     // mask. If not null, assumed to be fp16. nq x nk elements
                            [[maybe_unused]] float         scale,    // scale applied before softmax
                            [[maybe_unused]] float         softcap,  // if > 0, a "soft-cap" operation is applied before softmax
                            [[maybe_unused]] float       * qkv,      // v*softmax(scale*(k*q))
                            [[maybe_unused]] void * work_buffer, [[maybe_unused]] barrier_t barrier, [[maybe_unused]] void * barrier_data,
                            [[maybe_unused]] int ith, [[maybe_unused]] int nth, [[maybe_unused]] int n_swa, [[maybe_unused]] ggml_tensor * indexer) {
    return false;
}

#endif

