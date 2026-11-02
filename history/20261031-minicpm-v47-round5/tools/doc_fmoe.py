#!/usr/bin/env python3
# record the fused MoE tail results in PROGRESS.md and FA-BOUNDARY.md
import pathlib
root = pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round5')

p = root / 'PROGRESS.md'
s = p.read_text()
old = 'GLIBC_TUNABLES=glibc.malloc.hugetlb=1 IK_R5_DYN=1 IK_R5_FA=1 IK_R5_HUGE=1 \\\n'
assert s.count(old) == 1
s = s.replace(old, 'GLIBC_TUNABLES=glibc.malloc.hugetlb=1 IK_R5_DYN=1 IK_R5_FA=1 IK_R5_HUGE=1 IK_R5_FMOE=1 \\\n')
s += '''
| 1caa9665 | IK_R5_FMOE=1 | fused MoE tail at decode: routed down + weighted sum + residual + shared-expert down + gate + add in one node | bit-exact; mirror 2.8K +4.4%, see below |

## Fused MoE tail (IK_R5_FMOE, commit 1caa9665)
Per MoE layer the decode graph had 8 nodes after the routed up/gate (routed down MUL_MAT_ID, MUL_MULTI_ADD, ADD residual,
shared-expert gate MUL_MAT, shared FUSED_UP_GATE, shared down MUL_MAT, FUSED_MUL_UNARY, ADD). The executor now matches that
pattern and runs it as one node: shared gate and up/gate first, then every thread owns a 16-row slice of the 2048 output rows
and computes the 8 routed down rows + the shared down rows (node-local weights) and the whole combine. 8 barriers -> 5,
280 fewer nodes per token, no core gathers other cores' partials.
- Safety: the graph allocator reuses freed buffers, so in the decode graph the shared gate/up outputs can sit on top of the
  routed inputs the original order had finished with. The fused path therefore quantizes par and snapshots ids, routing
  weights and its residual slice before running them; after the last internal barrier no graph tensor is read. It only
  applies when the node pattern, shapes and work-buffer size check out (prompt graphs and the 256-expert warmup fall through).
  IK_R5_FMOE=2 also checks that no later node reads a skipped node (0 of 200 patterns); IK_R5_FMOE=3 prints misses.
- Correctness (runs/41-fmoe-probs): top-5 probabilities of 32 tokens bitwise identical to the unfused path, single socket and
  mirror, 2.8K and 9.4K.
- Profile (runs/43 vs runs/40, cross-session): MoE tail 4.54 -> 4.00 ms/token single socket, 4.33 -> 3.65 mirrored.
- Speed (runs/44 + runs/45, 6 interleaved rounds each, 512 tokens):

| Config | 2.8K base | 2.8K fused | 9.4K base | 9.4K fused |
|---|---:|---:|---:|---:|
| s32 (node 1) | 32.97 | 33.77 (+2.4%, 3/6 pairs won) | 29.47 | 29.98 (+1.7%, 4/6) |
| m64 (mirror) | 37.64 | 39.28 (+4.4%, 6/6) | 32.76 | 33.01 (+0.8%, 3/6) |

  Single-socket runs were noisy this session (+-2 tok/s between rounds), so only the mirrored 2.8K gain is clear-cut.
  Dropping the barrier between the shared gate and up/gate (IK_R5_FMOE_B=0) was not better (runs/44).
'''
p.write_text(s)

p = root / 'FA-BOUNDARY.md'
s = p.read_text()
s += '''- Fused MoE tail (IK_R5_FMOE, bit-exact): routed down + weighted sum + residual + shared expert down + gate + add as one
  node with per-thread output-row ownership. Tail 4.54 -> 4.00 ms/token single socket, 4.33 -> 3.65 mirrored; decode +4.4%
  mirrored at 2.8K, +0.8..2.4% elsewhere (noisy). Details in PROGRESS.md.
'''
p.write_text(s)
print('docs updated')
