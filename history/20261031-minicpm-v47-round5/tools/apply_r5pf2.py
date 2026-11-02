#!/usr/bin/env python3
"""Round-5 patch 21: IK_R5_PF issues 2 prefetches per barrier spin iteration by default (was 4).

The profile with IK_R5_PF=1 (runs/81) shows the ~860 mostly-sync small nodes per token ~0.6 ms slower than without
prefetch: a waiting thread issues IK_R5_PF_LINES prefetches before each check of the barrier, and a prefetch can
stall on a full miss queue, so release is noticed later. 2 lines per iteration: +3.3% at 2.8K (4/4 rounds), +1.0% at
9.4K (3/4) over 4 (runs/83). Skipping prefetch for the cross-node leader was worse (runs/82, runs/83), so it stays.
"""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'ggml/src/ggml.c'
s = p.read_text()
for old, new in (('static int    r5pf_mode = -1, r5pf_lines = 4,', 'static int    r5pf_mode = -1, r5pf_lines = 2,'),
                 ('    int lines = 4, win = 32;\n', '    int lines = 2, win = 32;\n')):
    assert s.count(old) == 1, old
    s = s.replace(old, new)
p.write_text(s)
print('patched')
