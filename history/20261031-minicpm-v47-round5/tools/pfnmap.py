#!/usr/bin/env python3
"""Physical placement of a process's large mappings: pfnmap.py PID [min_MiB=200]   (run as root)

For each anonymous mapping >= min_MiB: physical address range of its present pages, a 4 GiB-bucket
histogram, and the fraction backed by huge pages (via /proc/kpageflags bit 22 THP / 17 HUGE).
"""
import struct, sys, collections
pid = int(sys.argv[1]); min_mib = int(sys.argv[2]) if len(sys.argv) > 2 else 200
PAGE = 4096
maps = []
for line in open(f'/proc/{pid}/maps'):
    f = line.split()
    a, b = (int(x, 16) for x in f[0].split('-'))
    if (b - a) >> 20 >= min_mib and len(f) < 6:
        maps.append((a, b))
pm = open(f'/proc/{pid}/pagemap', 'rb')
kf = open('/proc/kpageflags', 'rb')
for a, b in maps:
    hist = collections.Counter(); n = 0; thp = 0; lo = None; hi = 0
    step = 64 * PAGE  # sample every 256 KiB
    for va in range(a, b, step):
        pm.seek((va // PAGE) * 8)
        e = struct.unpack('<Q', pm.read(8))[0]
        if not (e >> 63) & 1:
            continue
        pfn = e & ((1 << 55) - 1)
        pa = pfn * PAGE
        n += 1; hist[pa >> 32] += 1
        lo = pa if lo is None else min(lo, pa); hi = max(hi, pa)
        kf.seek(pfn * 8)
        fl = struct.unpack('<Q', kf.read(8))[0]
        if (fl >> 22) & 1 or (fl >> 17) & 1:
            thp += 1
    if n == 0:
        print(f'{a:x}-{b:x} {(b-a)>>20} MiB: no present pages'); continue
    buckets = ' '.join(f'{k*4}-{k*4+4}G:{100*v/n:.0f}%' for k, v in sorted(hist.items()))
    print(f'{a:x}-{b:x} {(b-a)>>20:6d} MiB  sampled {n}  THP {100*thp/n:5.1f}%  phys {lo/2**30:7.2f}-{hi/2**30:7.2f} GiB  [{buckets}]')
