#!/usr/bin/env python3
"""Which mappings take the page faults recorded by r5bench R5_PERF_RECORD=page-faults: faultmap.py MAPS PERF_DATA

Runs `perf script -F addr,tid` on the recording, buckets fault addresses by the mapping in the /proc/PID/maps
snapshot, and prints mappings by fault count with their size, plus the distinct 2 MiB regions touched.
"""
import bisect, collections, subprocess, sys
maps = []
for line in open(sys.argv[1]):
    f = line.split()
    a, b = (int(x, 16) for x in f[0].split('-'))
    maps.append((a, b, f[1], f[5] if len(f) > 5 else '[anon]'))
maps.sort()
starts = [m[0] for m in maps]
out = subprocess.run(['sudo', '-n', 'perf', 'script', '-i', sys.argv[2], '-F', 'tid,addr'],
                     capture_output=True, text=True).stdout
cnt = collections.Counter(); regions = collections.defaultdict(set); tids = collections.defaultdict(set); n = 0
for line in out.splitlines():
    f = line.split()
    if len(f) < 2:
        continue
    tid, addr = int(f[0]), int(f[1], 16)
    n += 1
    k = bisect.bisect_right(starts, addr) - 1
    key = k if k >= 0 and maps[k][0] <= addr < maps[k][1] else -1
    cnt[key] += 1; regions[key].add(addr >> 21); tids[key].add(tid)
print(f'{n} faults')
for key, c in cnt.most_common(12):
    if key < 0:
        print(f'{c:7d}  (outside the maps snapshot)'); continue
    a, b, perm, name = maps[key]
    print(f'{c:7d}  {a:x}-{b:x} {(b - a)/2**20:10.1f} MiB {perm} {name}  2MiB-regions {len(regions[key])}  threads {len(tids[key])}')
