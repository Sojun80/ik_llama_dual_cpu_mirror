#!/usr/bin/env python3
"""Round-5 patch 12: two-pass r5fa partial (env IK_R5_FA_2PASS=1).

In-model measurement (IK_R5_FA_STREAMTEST): reading K and V rows interleaved collapses DRAM
throughput to ~0.45 bytes/tick per thread on a 4-thread CCX, while reading the same rows as one
sequential K run followed by one V run reaches ~1.7-1.8. The online (block-interleaved) kernel
alternates K and V every 32 positions and its prefetches alternate per row.

Two passes per thread over its position range:
  A: stream K rows only, compute all scores S[h][j] (4-head x 2-position tiles, all KV heads per row)
  B: mask + exact softmax per head over the whole range (max known, no running rescale)
  C: stream V rows only in 32-row blocks, O[h] += p[h][j] * v_j
Each pass prefetches only its own stream, IK_R5_FA_PFB blocks ahead.
"""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'ggml/src/iqk/iqk_flash_attn.cpp'
s = p.read_text()
assert 'partial2pass' not in s

KERNEL = r'''
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

static bool run('''

anchor = '\nstatic bool run('
assert s.count(anchor) == 1
s = s.replace(anchor, KERNEL, 1)

# dispatch: the first (timed) partial call in run()
old = '''    const uint64_t t0 = dbg ? __rdtsc() : 0;
    partial(j0, j1, nek2, rk, (const char *)q, nbq2, (const char *)k, stride_k, nbk2, (const char *)v, stride_v, nbv2,
            mk, scale, mine, mine + neq2, mine + 2*neq2);'''
new = '''    static const bool two_pass = [] { auto e = getenv("IK_R5_FA_2PASS"); return e && atoi(e) > 0; }();
    const uint64_t t0 = dbg ? __rdtsc() : 0;
    (two_pass ? partial2pass : partial)(j0, j1, nek2, rk, (const char *)q, nbq2, (const char *)k, stride_k, nbk2,
            (const char *)v, stride_v, nbv2, mk, scale, mine, mine + neq2, mine + 2*neq2);'''
assert s.count(old) == 1
s = s.replace(old, new)
if '#include <vector>' not in s:
    s = s.replace('#include <cstring>', '#include <cstring>\n#include <vector>', 1)
p.write_text(s)
print('patched')
