#!/usr/bin/env python3
# record the router fusion (IK_R5_FMOE_UG=2, neutral, opt-in) in PROGRESS.md and FA-BOUNDARY.md
import pathlib
root = pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round5')

p = root / 'PROGRESS.md'
s = p.read_text()
row = '| c878b8ab | IK_R5_FMOE_UG=1 |'
a = s.index(row)
a = s.index('\n', a) + 1
s = s[:a] + ('| 96a11658 | IK_R5_FMOE_UG=2 | router (norm, logits, top-k, weight normalization) folded into the fused MoE node too '
             '| bit-exact; no measurable gain, opt-in only |\n') + s[a:]
s += '''
## Router in the fused MoE node (IK_R5_FMOE_UG=2, commit 96a11658) - neutral, not recommended
The 12 router nodes before the routed up/gate (FUSED_RMS_NORM, logits MUL_MAT, executor-fused SOFT_MAX..GET_ROWS,
executor-fused SUM_ROWS+DIV, reshapes) join the fused node: every thread computes the normed input itself
(ggml_compute_forward_fused_rms_norm on a private copy), the 256 logits rows are split over the threads, and after one
barrier every thread runs iqk_topk_moe and iqk_sumrows_div on private copies. Barriers from the norm to l_out 8 -> 4,
nodes per graph 863 -> 703. Only used when the MoE expert prefetch is off (the hook reads the graph's ids).
- Correctness (runs/51-fmrt-probs): bitwise identical to IK_R5_FMOE_UG=1, single socket and mirror, 2.8K and 9.4K;
  consumer check 0 external reads.
- Profile (runs/52-fmrt-prof, same session), norm .. l_out per token: mirrored 6.71 -> 6.67 ms, single socket
  9.19 -> 8.91 ms. Mean per-thread busy time rose (5.34 -> 6.08 ms mirrored): the repeated per-thread norm and top-k
  cost about what the removed barriers saved. The router logits matmul (2 MB of f32 weights per layer, ~19 us)
  dominates the router either way.
- Speed (runs/53-fmrt-bench, 4 interleaved rounds, 512 tokens), UG=1 -> UG=2: s32 2.8K 34.79 -> 33.77,
  9.4K 30.28 -> 30.56; m64 2.8K 40.56 -> 40.62, 9.4K 34.54 -> 34.62. Every config won exactly 2 of 4 pairs,
  always the one that ran second in its round, so this run also shows a run-order bias of ~1-2 tok/s that the
  ABBA order cancels.
- Kept as an opt-in for later work (e.g. sharing the top-k per CCX instead of per thread); recommended command unchanged.
'''
p.write_text(s)

p = root / 'FA-BOUNDARY.md'
s = p.read_text()
s += '''- Router folded into the fused MoE node as well (IK_R5_FMOE_UG=2, bit-exact): 8 -> 4 barriers per MoE layer but no
  measurable decode gain; the per-thread repeated norm/top-k offsets the saved barriers and the f32 router logits
  matmul dominates. Opt-in only; details in PROGRESS.md.
'''
p.write_text(s)
print('docs updated')
