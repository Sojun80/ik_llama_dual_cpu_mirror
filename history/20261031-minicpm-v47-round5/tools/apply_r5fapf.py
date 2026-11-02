#!/usr/bin/env python3
"""Round-5 patch 10: deeper KV software prefetch in r5fa::partial (env IK_R5_FA_PFB, default 3 blocks).

In-model the KV rows always come from DRAM (the weight stream evicts them between tokens) and each
thread streams only ~100-300 KB of K and V per call, so a one-block (32 position) prefetch leaves
the first block as demand misses and too few lines in flight. Now: an upfront burst covering the
first PFB blocks, then each block prefetches the block PFB ahead.
"""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'ggml/src/iqk/iqk_flash_attn.cpp'
s = p.read_text()
assert 'IK_R5_FA_PFB' not in s

def sub(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)

sub('''    const long span_v = (nkvh - 1)*nbv2 + 2*D;
''', '''    const long span_v = (nkvh - 1)*nbv2 + 2*D;
    static const int pfb = [] { auto e = getenv("IK_R5_FA_PFB"); int v = e ? atoi(e) : 3; return v < 1 ? 1 : v; }();
    // upfront burst: the first pfb blocks (the first one is needed right away)
    for (int j = j0; j < std::min(j1, j0 + pfb*JB); ++j) {
        const char * kr = k + (long)j*stride_k;
        const char * vr = v + (long)j*stride_v;
        for (long o = 0; o < span_k; o += 64) _mm_prefetch(kr + o, _MM_HINT_T1);
        for (long o = 0; o < span_v; o += 64) _mm_prefetch(vr + o, _MM_HINT_T1);
    }
''')
sub('''        const int jn = jb + JB, nn = std::max(0, std::min(JB, j1 - jn));''',
    '''        const int jn = jb + pfb*JB, nn = std::max(0, std::min(JB, j1 - jn));''')
p.write_text(s)
print('patched')
