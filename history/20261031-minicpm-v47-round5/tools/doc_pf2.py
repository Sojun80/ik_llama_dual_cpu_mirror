#!/usr/bin/env python3
# Record the 2-line prefetch default (2026-11-01, runs/82-84) in PROGRESS.md
import pathlib, sys
root = pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round5')
c = sys.argv[1]
p = root / 'PROGRESS.md'
s = p.read_text()
assert 'runs/83' not in s
row = '| IK_R5_PF_FA=1 |'
a = s.index(row)
a = s.index('\n', a) + 1
s = s[:a] + (f'| {c} | IK_R5_PF_LINES default 2 | 2 prefetches per barrier spin iteration instead of 4 | bit-exact (runs/84); '
             '+3.3% 2.8K, +1.0% 9.4K (runs/83) |\n') + s[a:]
s += f'''
## Trimming the prefetch cost (2026-11-01): IK_R5_PF_LINES default 2, commit {c}
The profile (runs/81) put the prefetch cost in the small nodes' barrier waits. First guess: the cross-node leader
(the last thread on a node, which releases its node's followers) noticing the release late. A build where that leader
only pauses (IK_R5_PF_LEADER, not kept) was tested together with fewer prefetches per spin iteration.

| tok/s, all with IK_R5_PF=1 | 2.8K | 9.4K |
|---|---:|---:|
| runs/82 (3 rounds): 4 lines, leader prefetches (previous default) | 44.47 | 39.30 |
| 4 lines, leader only pauses | 43.73 | 38.71 |
| 2 lines, leader only pauses | 45.37 | 40.12 |
| 1 line, leader only pauses | 45.34 | 39.79 |
| runs/83 (4 rounds): 4 lines, leader prefetches (previous default) | 43.52 | 39.53 |
| **2 lines, leader prefetches** | **44.94** (+3.3%, won 4/4) | **39.93** (+1.0%, won 3/4) |
| 2 lines, leader only pauses | 43.90 | 39.20 |

The leader's prefetching helps, so the leader change was dropped; the default is now 2 lines per iteration
(tools/apply_r5pf2.py). Every config drifted ~+2 tok/s over runs/82, so only within-run comparisons count.
runs/84: completion_probabilities with the new default bitwise identical to runs/77's no-prefetch reference at 2.8K
and 9.4K.
'''
p.write_text(s)
print('docs updated')
