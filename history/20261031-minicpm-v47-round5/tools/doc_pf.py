#!/usr/bin/env python3
# KV prefetch default off; record the barrier-wait prefetch (2026-11-01) in PROGRESS.md and the recommended command
import pathlib, sys
root = pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round5')
c_pf, c_fa = sys.argv[1], sys.argv[2]

p = root / 'PROGRESS.md'
s = p.read_text()
assert 'IK_R5_PF=1' not in s
old = 'GLIBC_TUNABLES=glibc.malloc.hugetlb=1 IK_R5_DYN=1 IK_R5_FA=1 IK_R5_HUGE=1 IK_R5_FMOE=1 IK_R5_FMOE_UG=1 \\\n'
assert s.count(old) == 1
s = s.replace(old, 'GLIBC_TUNABLES=glibc.malloc.hugetlb=1 IK_R5_DYN=1 IK_R5_FA=1 IK_R5_HUGE=1 IK_R5_FMOE=1 IK_R5_FMOE_UG=1 \\\n'
                   'IK_R5_NONUMAB=1 IK_R5_PF=1 \\\n')
row = '| 0520716c | IK_R5_FMOE_LOCAL=1 |'
a = s.index(row)
a = s.index('\n', a) + 1
s = s[:a] + (f'| {c_pf} | IK_R5_PF=1 | prefetch each thread\'s upcoming weights (static rows of the next dense matmuls, its MoE '
             'down rows once the experts are known) while it spins in a barrier | bit-exact; mirror decode +4.5% 2.8K, '
             '+7.8% 9.4K (runs/78) |\n'
             f'| {c_fa} | IK_R5_PF_FA=1 | also prefetch the thread\'s K/V positions of the next decode FA (opt-in) | '
             'neutral (runs/79) |\n') + s[a:]
s += f'''
## Prefetch while waiting at barriers (2026-11-01): IK_R5_PF=1, commits {c_pf} and {c_fa}
Profile (runs/49, m64 + IK_R5_FMOE_UG=1, 2.8K): of ~22.1 ms per token each thread is idle ~5.5 ms on average, ~1.07 ms
of it in the delta-net step where few threads have work. IK_R5_DYN=1 already gives every thread a fixed share (75%) of
the rows of each single-token dense matmul before the dynamic remainder, so its next reads are known in advance.
tools/apply_r5pf.py: before each node barrier, every thread queues its static rows of the next dense MUL_MAT nodes
(lookahead IK_R5_PF_WIN=32 nodes, up to IK_R5_PF_BUDGET=256 KiB not yet prefetched) from its node-local mirror copy;
r5_fmug replaces the queue with the thread's down rows (8 routed experts + shared, its output slice) as soon as the
experts are known. Every spin iteration of the NUMA barriers issues IK_R5_PF_LINES=4 prefetcht1 (L2) before its
pause. No result changes (runs/77: completion_probabilities bitwise identical at 2.8K and 9.4K).

| config (runs/78, ABBA, 4 rounds, casebook stopped) | 2.8K | 9.4K |
|---|---:|---:|
| m64 recommended so far | 41.47 | 35.40 |
| + IK_R5_PF=1 | **43.33** (+4.5%, won all 4 rounds) | **38.17** (+7.8%, won all 4 rounds) |
| + IK_R5_PF=1, 8 lines, 384 KiB budget | 43.08 | 37.53 |

Follow-ups (runs/79, ABBA, 4 rounds, all with IK_R5_PF=1): prefetching each thread's K/V positions of the next decode
FA (tools/apply_r5pffa.py, IK_R5_PF_FA=1) 43.86 -> 44.14 at 2.8K, 37.62 -> 37.64 at 9.4K: noise, so opt-in. A larger
static share (IK_R5_DYN_STATIC=90) is not better (43.97 / 37.01).
'''
p.write_text(s)
print('docs updated')
