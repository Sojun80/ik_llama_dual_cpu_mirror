#!/usr/bin/env python3
"""Round-5 diagnostic: IK_R5_FA_TWICE=1 runs r5fa::partial a second time right after the first
(same inputs, idempotent outputs) and reports per-thread ticks of pass 1 vs pass 2 every 640 calls."""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'ggml/src/iqk/iqk_flash_attn.cpp'
s = p.read_text()
assert 'IK_R5_FA_TWICE' not in s
old = '''    const uint64_t t0 = dbg ? __rdtsc() : 0;
    partial(j0, j1, nek2, rk, (const char *)q, nbq2, (const char *)k, stride_k, nbk2, (const char *)v, stride_v, nbv2,
            mk, scale, mine, mine + neq2, mine + 2*neq2);
    const uint64_t t1 = dbg ? __rdtsc() : 0;
'''
new = '''    const uint64_t t0 = dbg ? __rdtsc() : 0;
    partial(j0, j1, nek2, rk, (const char *)q, nbq2, (const char *)k, stride_k, nbk2, (const char *)v, stride_v, nbv2,
            mk, scale, mine, mine + neq2, mine + 2*neq2);
    const uint64_t t1 = dbg ? __rdtsc() : 0;
    static const bool twice = [] { auto e = getenv("IK_R5_FA_TWICE"); return e && atoi(e) > 0; }();
    if (twice) {
        static thread_local uint64_t p1, p2, nc;
        const uint64_t a = __rdtsc();
        partial(j0, j1, nek2, rk, (const char *)q, nbq2, (const char *)k, stride_k, nbk2, (const char *)v, stride_v, nbv2,
                mk, scale, mine, mine + neq2, mine + 2*neq2);
        p2 += __rdtsc() - a; p1 += t1 - t0;
        if (++nc % 640 == 0) {
            fprintf(stderr, "r5fa-twice ith %d positions %d: pass1 %.0f  pass2 %.0f ticks/call\\n", ith, j1 - j0, p1/640.0, p2/640.0);
            p1 = p2 = 0;
        }
    }
'''
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s)
print('patched')
