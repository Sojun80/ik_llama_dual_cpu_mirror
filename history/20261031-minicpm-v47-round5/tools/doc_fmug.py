#!/usr/bin/env python3
# record the routed + shared up/gate fusion (IK_R5_FMOE_UG) in PROGRESS.md and FA-BOUNDARY.md
import pathlib
root = pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round5')

p = root / 'PROGRESS.md'
s = p.read_text()
old = 'GLIBC_TUNABLES=glibc.malloc.hugetlb=1 IK_R5_DYN=1 IK_R5_FA=1 IK_R5_HUGE=1 IK_R5_FMOE=1 \\\n'
assert s.count(old) == 1
s = s.replace(old, 'GLIBC_TUNABLES=glibc.malloc.hugetlb=1 IK_R5_DYN=1 IK_R5_FA=1 IK_R5_HUGE=1 IK_R5_FMOE=1 IK_R5_FMOE_UG=1 \\\n')
row = '| 1caa9665 | IK_R5_FMOE=1 |'
a = s.index(row)
a = s.index('\n', a) + 1
s = s[:a] + ('| c878b8ab | IK_R5_FMOE_UG=1 | routed + shared up/gate folded into the fused MoE tail (9 nodes -> 1, 7 barriers -> 4) '
             '| bit-exact; mirror +4.8% 2.8K, +4.0% 9.4K, see below |\n') + s[a:]
s += '''
## Routed + shared up/gate in the fused MoE tail (IK_R5_FMOE_UG, commit c878b8ab)
Needs IK_R5_FMOE=1. The match now starts at the routed MOE_FUSED_UP_GATE node, so routed up/gate .. l_out (9 nodes)
run as one node in four phases:
  P0 one thread quantizes the FFN input, another computes the shared-expert gate logit (same iqk_mul_mat_4d call as the
     MUL_MAT node, into the work buffer); every thread snapshots ids, routing weights and its residual slice | barrier
  P1 one chunk pool of 8 x 16 routed up/gate chunks + 16 shared up/gate chunks (32 rows each, IK_R5_FMOE_CR), same
     iqk_moe_fused_up_gate kernel and row mapping as the original nodes, NUMA-aware claiming as before | barrier
  P2 quantize the 8 routed rows and the shared row, one row per thread | barrier
  P3 per-thread output rows: routed + shared down rows and the combine (unchanged from IK_R5_FMOE)
Nothing is written to a graph tensor before l_out, so compute-buffer reuse cannot bite. Barriers from the routed
up/gate to l_out: 7 -> 4; the shared up/gate no longer runs as its own unbalanced phase.
- Gotchas found on the way: the routed node reads a reshape view of the normed input (accept a zero-offset view);
  and the activation types here are Q8_2_X4 / Q8_0_X4, which interleave blocks, so rows must be quantized whole
  (a block-wise first attempt produced NaN logits).
- Correctness (runs/46-fmug-probs): top-5 probabilities of 32 tokens bitwise identical to IK_R5_FMOE=1, single socket
  and mirror, 2.8K and 9.4K; consumer check (IK_R5_FMOE=2) 0 external reads in 200 patterns.
- Profile (runs/49-fmug-prof, same session), routed up/gate .. l_out per token: mirrored 6.19 -> 5.36 ms,
  single socket 8.16 -> 7.77 ms.
- Speed (runs/48-fmug-bench, 4 interleaved rounds, 512 tokens):

| Config | 2.8K FMOE | 2.8K +UG | 9.4K FMOE | 9.4K +UG |
|---|---:|---:|---:|---:|
| s32 (node 1) | 33.96 | 34.00 (+0.1%, 2/4 pairs won) | 30.11 | 30.65 (+1.8%, 2/4) |
| m64 (mirror) | 40.62 | 42.58 (+4.8%, 4/4) | 34.66 | 36.04 (+4.0%, 4/4) |

  Single-socket runs stayed noisy (+-2 tok/s between rounds); the mirrored gain is consistent.
'''
p.write_text(s)

p = root / 'FA-BOUNDARY.md'
s = p.read_text()
s += '''- Routed + shared up/gate folded into the fused tail (IK_R5_FMOE_UG, bit-exact): 9 nodes and 7 barriers -> 1 node and
  4 barriers per MoE layer. Mirrored decode +4.8% at 2.8K and +4.0% at 9.4K over IK_R5_FMOE alone; single socket within
  noise. Details in PROGRESS.md.
'''
p.write_text(s)
print('docs updated')
