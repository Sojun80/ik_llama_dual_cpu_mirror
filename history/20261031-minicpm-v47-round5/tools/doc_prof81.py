#!/usr/bin/env python3
# Record the decode profile with IK_R5_PF=1 (runs/81, 2026-11-01) in PROGRESS.md
import pathlib
root = pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round5')
p = root / 'PROGRESS.md'
s = p.read_text()
assert 'runs/81' not in s
s += '''
## Decode profile with prefetch (2026-11-01, runs/81-pf-prof)
build-mirror3 (87c45b2d) has the r5prof profiler; IK_R5_PROF via `r5bench.py --prof`, 1 round per config, m64
recommended settings with and without IK_R5_PF=1. Comparison script: tools/profcmp.py (idle = wall - mean thread
compute per node class, us per token). The profiler slows decode (pf 37.2 / 36.9 tok/s). The two 2.8K runs ran first
and their output head is disturbed (7.7 / 4.4 ms instead of ~2 ms, as in runs/49 and the 9.4K runs), so read
the 9.4K pair and 2.8K without the output head.

| group (9.4K, us per token) | wall m64 | wall +PF | idle m64 | idle +PF |
|---|---:|---:|---:|---:|
| dense MUL_MAT (without output head) | 8144 | **5684** | 1685 | 1220 |
| MoE up/gate/down (fused) | 5957 | 5940 | 754 | 840 |
| flash attention | 4260 | 4373 | 95 | 131 |
| delta-net | 2415 | 2364 | 1454 | 1381 |
| output head | 2028 | 1839 | 89 | 46 |
| all other nodes (norms, adds, copies, softplus, ...) | 3323 | 3909 | 2593 | 3134 |
| per token | 26137 | 24121 | | |

- The prefetch gain is in the dense matmuls (-2.5 ms at 9.4K, -1.5 ms at 2.8K); the MoE step barely changes.
- Cost: the small nodes get ~0.6 ms slower with IK_R5_PF=1 (same at 2.8K: 3196 -> 3931 wall). They are almost all
  sync time, and a waiting thread now issues 4 prefetches before each check of the barrier; prefetches can stall
  when the miss queue is full, so release is noticed later.
- Remaining idle windows, largest first: small nodes ~3.1 ms (barrier-bound: ~860 nodes per token), delta-net
  ~1.4 ms (max - mean compute 1.34 ms: the kernel keeps only some threads busy), dense matmul ~1.2 ms,
  MoE ~0.8 ms.
'''
p.write_text(s)
print('docs updated')
