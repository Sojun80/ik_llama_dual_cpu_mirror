#!/usr/bin/env python3
# Record the delta-net split attempt and the cluster-paired CPU order (2026-11-01, runs/85-94) in PROGRESS.md
import pathlib
root = pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round5')
p = root / 'PROGRESS.md'
s = p.read_text()
assert 'runs/92' not in s
old = '--cpu-range $(cat tools/m64.txt)'
assert s.count(old) == 1, s.count(old)
s = s.replace(old, '--cpu-range $(cat tools/m64pair.txt)')
s += '''
## Delta-net: splitting heads over two threads (2026-11-01, not kept) and a better CPU order (kept)
The fused delta-net step at decode gives each of the 32 heads (head_dim 128, 64 KiB of f32 state) to one thread, so
32 of 64 threads idle there (~1.4 ms per token, runs/81). Within a head every value row is independent, so
IK_R5_DN_SPLIT=1 split each head's rows over two adjacent threads (same NUMA node), bit-exact.

- Runtime row bounds (first version): bit-exact (runs/85) but decode -12.6% at 2.8K, -10.2% at 9.4K (runs/86); the
  profile shows the kernel itself 2.5x slower (~74 -> ~200 us per layer, runs/87).
- Constant 64-row loops (second version, pointers offset to the part's first row): bit-exact (runs/89), still 2.4x
  slower in the kernel (runs/90).
- Cause: the recurrent state is only 32-byte aligned: 30 of 40 printed state pointers were 32 mod 64 (temporary debug
  build, runs/91). A 512-byte state column then splits two cache lines between the halves (the midpoint and the column
  edge), whatever the split point, and the partner threads of the m64 order are on different CCDs, so those lines bounce
  over the IO die. Microbenchmark tools/dnbench.cpp (runs/88, node-1 cores, us per layer of 16 heads): one thread
  whole heads 202; two halves, constant loops, aligned state: 130 same CCX / 133 other CCD; state 32 B off alignment:
  164 same CCX / 235 other CCD.
- tools/m64pair.txt: the same 64 CPUs, reordered so threads 2h and 2h+1 share a CCX (node 0: 16..47 in order; node 1:
  14 of 16 pairs, the 4 single-CPU CCXs of the m64 set pair across CCDs). runs/92 (ABBA, 3 rounds):

| tok/s | 2.8K | 9.4K |
|---|---:|---:|
| m64 order, IK_R5_PF=1 | 44.21 | 39.18 |
| **m64pair order** | **45.40** (+2.7%, won 3/3) | **40.29** (+2.8%, won 3/3) |
| m64pair order + IK_R5_DN_SPLIT=1 | 44.99 | 39.72 |

The split no longer loses much with paired threads but does not win, so it was dropped (tools/apply_r5dnsplit.NOT-KEPT.py,
not in ik-src). The order alone is bitwise identical to runs/77 at 2.8K and 9.4K (runs/93). Its gain is not in the
delta-net step (1982 -> 1991 us per token at 9.4K, runs/94) but in the MoE step (5758 -> 5532) and the smaller
matmuls (ffn_moe_logits, linear_attn_out): ~0.57 ms per token in all. Likely neighbouring threads reading neighbouring
rows now share an L3; not verified. The recommended command now uses tools/m64pair.txt (batch uses its first 48
entries: the same 32 node-0 CPUs plus a different 16 of node 1; prompt speed unchanged or slightly higher).
'''
p.write_text(s)
print('docs updated')
