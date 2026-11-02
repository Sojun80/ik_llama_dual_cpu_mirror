#!/usr/bin/env python3
"""Round-5 patch (diagnostic): IK_R5_FA_SHADOW=1 reads K/V in the r5 decode FA from a per-thread private copy.

The decode FA reads KV at a fixed ~20-25 GB/s per node whatever the number of readers (8..48 per node, runs/97 and
runs/98: per-reader speed scales inversely with the reader count), while the same threads stream weights at ~77 GB/s
per node and earlier in-kernel stream tests read a private cold buffer ~3x faster than their own KV rows
(FA-BOUNDARY.md). To separate "where/how the KV memory is" from "how the kernel reads", each thread keeps a private
copy of its K and V rows [j0, j1) per FA call site (keyed by the K/V pointers, i.e. layer and NUMA copy) in memory it
allocated and touched itself (2 MiB aligned, MADV_HUGEPAGE). Decode appends one row per token and never rewrites
older rows, so the copy is extended by the new rows while n_eff advances by exactly one per call at that site, and
refilled whenever its range start moves, it does not fit, or n_eff jumps (a new sequence). partial() then reads the
copy; the values are the same, so results are bit-identical. Diagnostic, off by default.
"""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'ggml/src/iqk/iqk_flash_attn.cpp'
s = p.read_text()
assert 'IK_R5_FA_SHADOW' not in s

old = '''static bool run(int neq2, long nbq2, int nek2, long nbk2, long nbv2, long nb1, int nek1,
'''
assert s.count(old) == 1
s = s.replace(old, '''// round-5 diagnostic (IK_R5_FA_SHADOW=1): per-thread private copy of the thread's K/V rows of one FA call site
struct Shadow { const char * k = nullptr; const char * v = nullptr; char * buf = nullptr; size_t cap = 0;
                int j0 = 0, j1 = 0, n_eff = 0; };
static bool shadow_enabled() {
    static const bool e = [] { auto s = getenv("IK_R5_FA_SHADOW"); return s && atoi(s) > 0; }();
    return e;
}
// private copies of K and V rows [j0, j1) (row bytes each, packed); false if no slot is left
static bool shadow_rows(const char * k, const char * v, long stride_k, long stride_v, long row, int j0, int j1,
                        int n_eff, const char ** kb_out, const char ** vb_out) {
    constexpr int kSites = 64;
    static thread_local Shadow sh[kSites];
    static thread_local int n_sh = 0;
    Shadow * e = nullptr;
    for (int i = 0; i < n_sh; ++i) if (sh[i].k == k && sh[i].v == v) { e = &sh[i]; break; }
    if (!e) {
        if (n_sh == kSites) return false;
        e = &sh[n_sh++]; e->k = k; e->v = v;
    }
    const size_t need = (size_t)(j1 - j0)*row;   // per K or V
    bool refill = !e->buf || e->j0 != j0 || n_eff != e->n_eff + 1 || j1 < e->j1;
    if (2*need > e->cap) {
        if (e->buf) free(e->buf);
        const size_t half = ((need + need/2) + (1u << 20) - 1) & ~(size_t)((1u << 20) - 1);
        e->cap = 2*half;
        e->buf = (char *)aligned_alloc(2u << 20, e->cap);
        madvise(e->buf, e->cap, MADV_HUGEPAGE);
        memset(e->buf, 0, e->cap);
        refill = true;
    }
    char * kb = e->buf, * vb = e->buf + e->cap/2;
    for (int j = refill ? j0 : e->j1; j < j1; ++j) {
        memcpy(kb + (size_t)(j - j0)*row, k + (long)j*stride_k, row);
        memcpy(vb + (size_t)(j - j0)*row, v + (long)j*stride_v, row);
    }
    e->j0 = j0; e->j1 = j1; e->n_eff = n_eff;
    *kb_out = kb; *vb_out = vb;
    return true;
}

''' + old)

old = '''    static const bool two_pass = [] { auto e = getenv("IK_R5_FA_2PASS"); return e && atoi(e) > 0; }();
    const uint64_t t0 = dbg ? __rdtsc() : 0;
    (two_pass ? partial2pass : partial)(j0, j1, nek2, rk, (const char *)q, nbq2, (const char *)k, stride_k, nbk2,
            (const char *)v, stride_v, nbv2, mk, scale, mine, mine + neq2, mine + 2*neq2);
'''
assert s.count(old) == 1
s = s.replace(old, '''    static const bool two_pass = [] { auto e = getenv("IK_R5_FA_2PASS"); return e && atoi(e) > 0; }();
    // round-5 diagnostic (IK_R5_FA_SHADOW=1): read this thread's rows from its private copy
    const char * kp = (const char *)k, * vp = (const char *)v;
    long sk = stride_k, sv = stride_v;
    const long row = (long)(nek2 - 1)*nbk2 + 2*D;   // bytes of one position over all KV heads
    if (shadow_enabled() && j1 > j0 && row == (long)(nek2 - 1)*nbv2 + 2*D) {
        const char * kb, * vb;
        if (shadow_rows((const char *)k, (const char *)v, stride_k, stride_v, row, j0, j1, n_eff, &kb, &vb)) {
            // partial() indexes rows from j0: rebase so that row j0 is the first private row
            kp = kb - (long)j0*row; vp = vb - (long)j0*row;
            sk = row; sv = row;
        }
    }
    const uint64_t t0 = dbg ? __rdtsc() : 0;
    (two_pass ? partial2pass : partial)(j0, j1, nek2, rk, (const char *)q, nbq2, kp, sk, nbk2,
            vp, sv, nbv2, mk, scale, mine, mine + neq2, mine + 2*neq2);
''')
p.write_text(s)
print('patched')
