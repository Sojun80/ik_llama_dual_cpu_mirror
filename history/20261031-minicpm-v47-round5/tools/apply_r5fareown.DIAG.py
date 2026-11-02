#!/usr/bin/env python3
"""Round-5 patch (diagnostic): IK_R5_FA_REOWN=1 makes each decode-FA thread the last writer of its own K/V rows.

IK_R5_FA_SHADOW (a per-thread private copy) makes the decode FA read 2.0-2.4x faster (runs/100, runs/101), while
writing each node's KV copy only from its own node does not help (IK_R5_KVLOCAL, runs/101). This tests the in-place
alternative: when a thread's rows [j0, j1) at an FA call site (keyed by the K/V pointers) are new to it, it re-stores
the first 8 bytes of every cache line of those K and V rows with the value just read, so the reader owns the lines.
The bytes never change (results bit-identical) and there is nothing to invalidate: a row rewritten later by the
graph is only slower to read until the range moves again.
"""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'ggml/src/iqk/iqk_flash_attn.cpp'
s = p.read_text()
assert 'IK_R5_FA_REOWN' not in s

old = '''static bool run(int neq2, long nbq2, int nek2, long nbk2, long nbv2, long nb1, int nek1,
'''
assert s.count(old) == 1
s = s.replace(old, '''// round-5 diagnostic (IK_R5_FA_REOWN=1): re-store one word per cache line of the rows this thread has not owned yet
static bool reown_enabled() {
    static const bool e = [] { auto s = getenv("IK_R5_FA_REOWN"); return s && atoi(s) > 0; }();
    return e;
}
static void reown_rows(const char * k, const char * v, long stride_k, long stride_v, long row, int j0, int j1) {
    constexpr int kSites = 64;
    struct Site { const char * k, * v; int j0, j1; };
    static thread_local Site st[kSites];
    static thread_local int n_st = 0;
    Site * e = nullptr;
    for (int i = 0; i < n_st; ++i) if (st[i].k == k && st[i].v == v) { e = &st[i]; break; }
    if (!e) {
        if (n_st == kSites) return;
        e = &st[n_st++]; *e = { k, v, 0, 0 };
    }
    // rows [e->j0, e->j1) were owned before; touch the rest of [j0, j1)
    auto touch = [&](int a, int b) {
        for (int j = a; j < b; ++j) {
            for (const char * base : { k + (long)j*stride_k, v + (long)j*stride_v }) {
                char * p0 = (char *)((uintptr_t)base & ~(uintptr_t)63);
                for (char * p = p0; p < base + row; p += 64) {
                    volatile uint64_t * w = (volatile uint64_t *)p;
                    *w = *w;
                }
            }
        }
    };
    if (j0 >= e->j1 || j1 <= e->j0) touch(j0, j1);
    else { if (j0 < e->j0) touch(j0, e->j0); if (j1 > e->j1) touch(e->j1, j1); }
    e->j0 = j0; e->j1 = j1;
}

''' + old)

old = '''    const uint64_t t0 = dbg ? __rdtsc() : 0;
    (two_pass ? partial2pass : partial)(j0, j1, nek2, rk, (const char *)q, nbq2, kp, sk, nbk2,
'''
assert s.count(old) == 1
s = s.replace(old, '''    if (reown_enabled() && j1 > j0 && kp == (const char *)k) {
        reown_rows((const char *)k, (const char *)v, stride_k, stride_v, row, j0, j1);
    }
''' + old)
p.write_text(s)
print('patched')
