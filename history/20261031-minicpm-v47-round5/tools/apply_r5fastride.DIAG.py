#!/usr/bin/env python3
"""Round-5 patch (diagnostic): IK_R5_FA_STRIDE=s lets only every s-th thread read K/V in the r5 decode FA.

runs/97: with 96 instead of 64 threads the decode FA gets slower (critical path 1.22M -> 1.52M ticks at 9.4K; ~6.1K ->
~11.7K ticks per position), i.e. KV throughput per node falls as readers are added. To see whether fewer readers
stream KV faster, threads with ith % s != 0 skip the read pass and leave an empty partial (m = -inf, which the merge
already skips); the positions are split over the nth/s readers. s = 1 (default) is the current kernel. Any s > 1
changes the partition and so the rounding of the merge: diagnostic, not bit-exact against s = 1.
"""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'ggml/src/iqk/iqk_flash_attn.cpp'
s = p.read_text()
assert 'IK_R5_FA_STRIDE' not in s
old = '''    // contiguous position ranges, multiples of 8
    const int per = ((n_eff + nth - 1)/nth + 7) & ~7;
    const int j0 = std::min(n_eff, ith*per), j1 = std::min(n_eff, j0 + per);
'''
assert s.count(old) == 1
s = s.replace(old, '''    // round-5 diagnostic (IK_R5_FA_STRIDE=s): only threads with ith % s == 0 read K/V; the others leave an empty partial
    static const int fa_stride = [] { auto e = getenv("IK_R5_FA_STRIDE"); int x = e ? atoi(e) : 1; return x < 1 ? 1 : x; }();
    const int str = nth % fa_stride == 0 ? fa_stride : 1;
    const int nrd = nth/str, ird = ith/str;
    const bool reader = ith % str == 0;
    // contiguous position ranges, multiples of 8
    const int per = ((n_eff + nrd - 1)/nrd + 7) & ~7;
    const int j0 = reader ? std::min(n_eff, ird*per) : n_eff, j1 = reader ? std::min(n_eff, j0 + per) : n_eff;
''')
p.write_text(s)
print('patched')
