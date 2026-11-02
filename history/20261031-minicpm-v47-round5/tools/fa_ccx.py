#!/usr/bin/env python3
"""Per-thread FA time vs CCX occupancy: fa_ccx.py SERVER_LOG CPULIST [active_threads]

Uses the last r5fa-phase line of every thread (IK_R5_FA_DEBUG=1): busy = setup+scores+softmax+V ticks per call.
Thread ith runs on CPULIST[ith]; CCX = cpu//4 (Zen 2, 4 cores per CCX). Prints mean busy and ticks per position
grouped by node and by how many of the active threads share the thread's CCX.
"""
import re, sys, collections
cpus = [int(x) for x in sys.argv[2].split(',')]
last = {}
for line in open(sys.argv[1]):
    m = re.match(r'r5fa-phase ith (\d+) \(positions (\d+)\): setup (\d+)\s+scores (\d+)\s+softmax (\d+)\s+V (\d+)', line)
    if m:
        last[int(m.group(1))] = tuple(int(x) for x in m.groups()[1:])
active = [i for i, v in last.items() if v[0] > 0]
occ = collections.Counter(cpus[i]//4 for i in active)
g = collections.defaultdict(list)
for i in active:
    pos, s, sc, sm, v = last[i]
    node = 0 if cpus[i] < 48 else 1
    g[(node, occ[cpus[i]//4])].append(((s + sc + sm + v), pos))
print(f'{len(active)} active threads')
for k in sorted(g):
    xs = g[k]
    busy = sum(b for b, _ in xs)/len(xs); per = sum(b/p for b, p in xs)/len(xs)
    print(f'node {k[0]}  {k[1]} active threads in the CCX: {len(xs):2d} threads  busy {busy:9.0f} ticks  {per:7.0f} ticks/position')
