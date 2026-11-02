#!/usr/bin/env python3
"""Per-mapping NUMA page placement of a process: numamap_summary.py PID [min_MiB=32]

Reads /proc/PID/numa_maps (needs the same user or root) and prints, for every mapping of at
least min_MiB resident, its policy and how many MiB sit on each node (huge pages counted with
their kernelpagesize).
"""
import re, sys
pid = sys.argv[1]
min_mib = float(sys.argv[2]) if len(sys.argv) > 2 else 32
rows = []
for line in open(f'/proc/{pid}/numa_maps'):
    f = line.split()
    addr, policy = f[0], f[1]
    kps = 4096
    nodes = {}
    for tok in f[2:]:
        m = re.match(r'kernelpagesize_kB=(\d+)', tok)
        if m:
            kps = int(m.group(1))*1024
    for tok in f[2:]:
        m = re.match(r'N(\d+)=(\d+)', tok)
        if m:
            nodes[int(m.group(1))] = int(m.group(2))
    mib = {n: c*kps/2**20 for n, c in nodes.items()}
    tot = sum(mib.values())
    if tot >= min_mib:
        rows.append((addr, policy, kps, tot, mib, ' '.join(t for t in f[2:] if t.startswith(('anon', 'file', 'heap', 'stack')))))
for addr, policy, kps, tot, mib, kind in rows:
    split = '  '.join(f'N{n}={v:9.1f}' for n, v in sorted(mib.items()))
    print(f'{addr} {policy:22s} page={kps//1024:5d}K total={tot:9.1f} MiB  {split}  {kind}')
