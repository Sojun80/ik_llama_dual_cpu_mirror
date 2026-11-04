#!/usr/bin/env python3
"""Flint patch 1: the r5 decode FA kernel (IK_R5_FA) handles Flint's attention.

Stock FA is 19% of a token at 2.8K and 33% at 9.4K on Flint (runs/04-prof-files). r5fa was restricted to head dim 256,
16 heads, no attention sinks, no sliding window. Now:
- head dim is a template parameter (128 or 256), up to 32 heads;
- sinks are applied at the merge, with the same formula as the generic GQA path;
- sliding-window layers: same bound as the generic path (the window lies in the last nblock*256 cache cells), then the
  leading masked cells are skipped; the window is split proportionally over all threads, so it is shared by both sockets;
- IK_R5_FA_SHADOW stays off for sliding-window layers: the window moves every token, a private copy would refill always.
MiniCPM (head dim 256, no sinks, no window) takes the same code path as before.
Like IK_R5_FA itself this changes reduction order, so results are not bitwise identical to ik's FA.
"""
import sys, pathlib, re
root = pathlib.Path(sys.argv[1])
p = root / 'ggml/src/iqk/iqk_flash_attn.cpp'
s = p.read_text()
assert 'r5fa_flint' not in s


def sub(old, new, n=1):
    global s
    assert s.count(old) == n, (old[:80], s.count(old))
    s = s.replace(old, new)


sub('''constexpr int D  = 256;
constexpr int JB = 32;
constexpr int kMaxHeads = 16;
''', '''// r5fa_flint: head dim is a template parameter of partial/partial2pass/run (128 Flint, 256 MiniCPM)
constexpr int JB = 32;
constexpr int kMaxHeads = 32;
''')
sub('''static inline size_t slot_floats(int neq2) { return (2*(size_t)neq2 + (size_t)neq2*D + 15) & ~(size_t)15; }''',
    '''static inline size_t slot_floats(int neq2, int D) { return (2*(size_t)neq2 + (size_t)neq2*D + 15) & ~(size_t)15; }''')
sub('''    if (Dk != D || Dv != D || neq1 != 1 || neq3 != 1 || nek3 != 1 || nev3 != 1) return false;
''', '''    if (Dk != Dv || (Dk != 128 && Dk != 256) || neq1 != 1 || neq3 != 1 || nek3 != 1 || nev3 != 1) return false;
''')
sub('''    return max_bias <= 0.0f && softcap == 0.0f && mask && !sinks && n_swa == 0 && !indexer;
''', '''    return max_bias <= 0.0f && softcap == 0.0f && mask && n_swa >= 0 && !indexer;
''')
sub('''                            float max_bias, float softcap, const void * mask, const void * sinks,
                            int n_swa, const void * indexer) {
    if (!enabled()) return false;
''', '''                            float max_bias, float softcap, const void * mask, const void * /*sinks*/,
                            int n_swa, const void * indexer) {
    if (!enabled()) return false;
''')
sub('''static void partial(int j0, int j1, int nkvh, int rk,
''', '''template <int D>
static void partial(int j0, int j1, int nkvh, int rk,
''')
sub('''static void partial2pass(int j0, int j1, int nkvh, int rk,
''', '''template <int D>
static void partial2pass(int j0, int j1, int nkvh, int rk,
''')
sub('''static bool run(int neq2, long nbq2, int nek2, long nbk2, long nbv2, long nb1, int nek1,
                int stride_k, int stride_v, const void * q, const void * k, const void * v, const void * mask,
                float scale, float * qkv, void * work, barrier_t barrier, void * barrier_data, int ith, int nth) {
''', '''template <int D>
static bool run(int neq2, long nbq2, int nek2, long nbk2, long nbv2, long nb1, int nek1,
                int stride_k, int stride_v, const void * q, const void * k, const void * v, const void * mask,
                const float * sinks, int n_swa,
                float scale, float * qkv, void * work, barrier_t barrier, void * barrier_data, int ith, int nth) {
''')
sub('''    const size_t slot = slot_floats(neq2);
    float * mine = (float *)work + (size_t)ith*slot;
    // contiguous position ranges, multiples of 8
    const int per = ((n_eff + nth - 1)/nth + 7) & ~7;
    const int j0 = std::min(n_eff, ith*per), j1 = std::min(n_eff, j0 + per);
''', '''    const size_t slot = slot_floats(neq2, D);
    float * mine = (float *)work + (size_t)ith*slot;
    // contiguous position ranges, multiples of 8
    int per = ((n_eff + nth - 1)/nth + 7) & ~7;
    int j0 = std::min(n_eff, ith*per), j1 = std::min(n_eff, j0 + per);
    if (n_swa > 0) {
        // same bound as the generic path: the window lies in the last nblock*256 cells
        constexpr int kMinBatch = 256;
        const int nblock = (2*kMinBatch + n_swa - 1)/kMinBatch;
        int first = std::max(0, nek1 - nblock*kMinBatch);
        while (first < n_eff && mk[first] == 0xFC00) ++first;
        const long n = n_eff - first;
        // proportional split: a window spans few positions and must not stay on the first socket's threads
        per = (int)(n/nth);
        j0 = first + (int)(ith*n/nth);
        j1 = first + (int)((ith + 1)*n/nth);
    }
''')
sub('''if (j1 > j0 && row == (long)(nek2 - 1)*nbv2 + 2*D && shadow_enabled()) {''',
    '''if (j1 > j0 && n_swa == 0 && row == (long)(nek2 - 1)*nbv2 + 2*D && shadow_enabled()) {''')
sub('''    (two_pass ? partial2pass : partial)(j0, j1, nek2, rk, (const char *)q, nbq2, kp, sk, nbk2,
''', '''    (two_pass ? partial2pass<D> : partial<D>)(j0, j1, nek2, rk, (const char *)q, nbq2, kp, sk, nbk2,
''')
sub('''        partial(j0, j1, nek2, rk, (const char *)q, nbq2, (const char *)k, stride_k, nbk2, (const char *)v, stride_v, nbv2,
''', '''        partial<D>(j0, j1, nek2, rk, (const char *)q, nbq2, (const char *)k, stride_k, nbk2, (const char *)v, stride_v, nbv2,
''')
sub('''        const __m256 vn = _mm256_set1_ps(L > 0 ? 1.0f/L : 0.0f);
        float * out = (float *)((char *)qkv + h*nb1) + d0;
''', '''        if (sinks) {
            // as in the generic GQA path: the sink joins the softmax as one more score
            const float s = sinks[h];
            if (s > M) {
                const float c = expf(M - s);
                const __m256 vc = _mm256_set1_ps(c);
                for (int i = 0; i < DC/8; ++i) acc[i] = _mm256_mul_ps(vc, acc[i]);
                L = L*c + 1;
            } else {
                L += expf(s - M);
            }
        }
        const __m256 vn = _mm256_set1_ps(L > 0 ? 1.0f/L : 0.0f);
        float * out = (float *)((char *)qkv + h*nb1) + d0;
''')
sub('''    if (r5fa::enabled() && Qr->type == GGML_TYPE_F32 && Kr->type == GGML_TYPE_F16 && Vr->type == GGML_TYPE_F16 &&
        Qr->ne[0] == r5fa::D && Vr->ne[0] == r5fa::D && Qr->ne[1] == 1 && Qr->ne[2] <= r5fa::kMaxHeads) {
        size = std::max(size, sizeof(float)*r5fa::slot_floats((int)Qr->ne[2])*nth);
    }
''', '''    if (r5fa::enabled() && Qr->type == GGML_TYPE_F32 && Kr->type == GGML_TYPE_F16 && Vr->type == GGML_TYPE_F16 &&
        (Qr->ne[0] == 128 || Qr->ne[0] == 256) && Vr->ne[0] == Qr->ne[0] && Qr->ne[1] == 1 && Qr->ne[2] <= r5fa::kMaxHeads) {
        size = std::max(size, sizeof(float)*r5fa::slot_floats((int)Qr->ne[2], (int)Qr->ne[0])*nth);
    }
''')
sub('''        return r5fa::run(neq2, nbq2, nek2, nbk2, nbv2, nb1, nek1, stride_k, stride_v, q, k, v, mask, scale, qkv,
                         work_buffer_in, barrier, barrier_data, ith, nth);
''', '''        auto run = Dk == 128 ? r5fa::run<128> : r5fa::run<256>;
        return run(neq2, nbq2, nek2, nbk2, nbv2, nb1, nek1, stride_k, stride_v, q, k, v, mask, (const float *)sinks, n_swa,
                   scale, qkv, work_buffer_in, barrier, barrier_data, ith, nth);
''')
p.write_text(s)
print('patched')
