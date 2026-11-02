#!/usr/bin/env python3
"""Round-5 diagnostic: IK_R5_FA_ROT=1 makes each FA thread start at an irregular offset inside its position range
(js = j0 + (ith*13 + 5*(ith/4)) % len) and wrap around: partial(js, j1) and partial(j0, js), merged with the online-softmax
rule. Tests whether all threads streaming K/V in lockstep at a fixed 0x4C000-byte spacing concentrate on a few memory
channels/banks. Changes the summation order (not bit-exact), diagnostic only."""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'ggml/src/iqk/iqk_flash_attn.cpp'
s = p.read_text()
assert 'IK_R5_FA_ROT' not in s
old = '''    const uint64_t t0 = dbg ? __rdtsc() : 0;
    (two_pass ? partial2pass : partial)(j0, j1, nek2, rk, (const char *)q, nbq2, (const char *)k, stride_k, nbk2,
            (const char *)v, stride_v, nbv2, mk, scale, mine, mine + neq2, mine + 2*neq2);
'''
assert s.count(old) == 1
s = s.replace(old, '''    const uint64_t t0 = dbg ? __rdtsc() : 0;
    static const bool rot = [] { auto e = getenv("IK_R5_FA_ROT"); return e && atoi(e) > 0; }();
    const int js = rot && j1 - j0 > 1 ? j0 + (ith*13 + 5*(ith/4)) % (j1 - j0) : j0;
    if (js > j0) {
        // diagnostic: rotated start; [js, j1) into this thread's slot, [j0, js) into a stack slot, then merge
        alignas(32) float tmp[2*kMaxHeads + kMaxHeads*D];
        partial(js, j1, nek2, rk, (const char *)q, nbq2, (const char *)k, stride_k, nbk2,
                (const char *)v, stride_v, nbv2, mk, scale, mine, mine + neq2, mine + 2*neq2);
        partial(j0, js, nek2, rk, (const char *)q, nbq2, (const char *)k, stride_k, nbk2,
                (const char *)v, stride_v, nbv2, mk, scale, tmp, tmp + neq2, tmp + 2*neq2);
        for (int h = 0; h < neq2; ++h) {
            const float ma = mine[h], mb = tmp[h], M = std::max(ma, mb);
            if (M == -INFINITY) continue;
            const float wa = ma == -INFINITY ? 0.0f : expf(ma - M), wb = mb == -INFINITY ? 0.0f : expf(mb - M);
            mine[h] = M;
            mine[neq2 + h] = wa*mine[neq2 + h] + wb*tmp[neq2 + h];
            float * oa = mine + 2*neq2 + (size_t)h*D;
            const float * ob = tmp + 2*neq2 + (size_t)h*D;
            for (int d = 0; d < D; ++d) oa[d] = wa*oa[d] + wb*ob[d];
        }
    } else
    (two_pass ? partial2pass : partial)(j0, j1, nek2, rk, (const char *)q, nbq2, (const char *)k, stride_k, nbk2,
            (const char *)v, stride_v, nbv2, mk, scale, mine, mine + neq2, mine + 2*neq2);
''')
p.write_text(s)
print('patched')
