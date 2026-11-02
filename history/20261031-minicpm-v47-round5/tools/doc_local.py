#!/usr/bin/env python3
# record the socket-local fused MoE step (IK_R5_FMOE_LOCAL, 2026-11-01) in PROGRESS.md
import pathlib, sys
root = pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round5')
commit = sys.argv[1]
p = root / 'PROGRESS.md'
s = p.read_text()
assert 'IK_R5_FMOE_LOCAL' not in s
row = '| 4ab938e6 | IK_R5_NONUMAB=1 |'
a = s.index(row)
a = s.index('\n', a) + 1
s = s[:a] + (f'| {commit} | IK_R5_FMOE_LOCAL=1 | socket-local fused MoE step: each node runs up/gate, quantization and down of '
             'its own experts with node-local barriers, one cross-node barrier per MoE layer instead of three (opt-in, '
             'default off) | bit-exact; decode ~1% slower (runs/76) |\n') + s[a:]
s += f'''
## Socket-local MoE step (2026-11-01): IK_R5_FMOE_LOCAL=1, commit {commit}, tools/apply_r5local.py
In r5_fmug (IK_R5_FMOE_UG=1) node k now owns the experts at positions [k*n/2, (k+1)*n/2) of the selected list and half
of the shared expert's up/gate rows. It quantizes its own input copy, runs its up/gate chunks from its own pool (no
cross-node stealing), quantizes its routed rows and computes their down rows for all 2048 outputs, with node-local
barriers between the steps. One full barrier follows, then every thread computes its slice of the shared down rows
and the unchanged combine. Cross-socket barriers per MoE layer: 1 instead of 3 (plus the executor's end-of-node one).
Bit-exact against the default path at 2.8K and 9.4K (runs/75, completion_probabilities identical).
Speed (runs/76, ABBA, 4 rounds, casebook stopped): 2.8K 41.25 -> 40.84 tok/s (slower in all 4 rounds), 9.4K
36.25 -> 36.05. The saved cross-socket barriers are worth less than the cross-node work stealing they replace: with a
fixed 4+4 split the slower node (node 0 reads memory more slowly in mirror mode, see FA-BOUNDARY.md) sets the pace,
and the three extra node-local barriers are not free. Left in as an option, off by default.
Note: m64 decoded 41.3 / 36.3 here and 43.0 / 38.0 in runs/69 with the same build and settings; run-to-run
variation across sessions is larger than within one ABBA run, so compare only within a run.
'''
p.write_text(s)
print('docs updated')
