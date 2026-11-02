#!/usr/bin/env python3
"""Round-5 patch 21: no barrier-wait prefetch for the cross-node leader (IK_R5_PF).

The profile with IK_R5_PF=1 (runs/81) shows the ~860 small nodes per token ~0.6 ms slower: they are almost all
sync time. In the hierarchical barrier the last thread to arrive on a node waits for the other node's leader and then
releases its node's followers, so any delay in noticing the global release (a prefetch stalling on a full miss queue)
delays every thread of that node. The leader has just finished computing and is on the critical path, so it now only
pauses; followers keep prefetching. IK_R5_PF_LEADER=1 restores the previous behaviour.
"""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'ggml/src/ggml.c'
s = p.read_text()
assert 'IK_R5_PF_LEADER' not in s

old = '''static int    r5pf_mode = -1, r5pf_lines = 4, r5pf_win = 32, r5pf_fa = 0;
'''
assert s.count(old) == 1
s = s.replace(old, '''static int    r5pf_mode = -1, r5pf_lines = 4, r5pf_win = 32, r5pf_fa = 0, r5pf_leader = 0;
''')
old = '''    r5pf_lines = lines; r5pf_win = win; r5pf_budget = budget;
'''
assert s.count(old) == 1
s = s.replace(old, old + '''    // the cross-node leader is on the critical path of every barrier: it only pauses unless IK_R5_PF_LEADER=1
    r5pf_leader = (e = getenv("IK_R5_PF_LEADER")) && atoi(e) > 0;
''')

old = '''            while (atomic_load(&g_numa_barrier.global_release) == g_old) {
                r5pf_spin();
            }
'''
assert s.count(old) == 1
s = s.replace(old, '''            while (atomic_load(&g_numa_barrier.global_release) == g_old) {
                if (r5pf_leader) r5pf_spin(); else ggml_numa_spin_pause();
            }
''')
p.write_text(s)
print('patched')
