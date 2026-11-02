#!/usr/bin/env python3
# thpmap.py PID: anonymous mappings >= 64 MiB of a process with their huge-page coverage and node placement
import sys, re
pid = sys.argv[1]
ALL = len(sys.argv) > 2
maps, cur = [], None
for line in open(f'/proc/{pid}/smaps'):
    m = re.match(r'^([0-9a-f]+)-([0-9a-f]+) (\S+) \S+ \S+ \S+\s*(.*)$', line)
    if m:
        cur = {'a': int(m.group(1), 16), 'b': int(m.group(2), 16), 'name': m.group(4).strip()}
        maps.append(cur)
        continue
    m = re.match(r'^(\w+):\s+(\d+) kB', line)
    if m and cur is not None:
        cur[m.group(1)] = int(m.group(2))
numa = {}
for line in open(f'/proc/{pid}/numa_maps'):
    f = line.split()
    numa[int(f[0], 16)] = ' '.join(x for x in f[1:] if x.startswith(('N0=', 'N1=', 'bind', 'default', 'local', 'prefer', 'interleave', 'kernelpagesize')))
for m in maps:
    size = (m['b'] - m['a']) >> 20
    if size < (1 if ALL else 64) or (m["name"] and not ALL):
        continue
    rss = m.get('Rss', 0) >> 10
    thp = m.get('AnonHugePages', 0) >> 10
    print(f"{m['a']:#x} {size:7d} MiB  rss {rss:7d} MiB  THP {thp:7d} MiB ({100*thp/max(rss,1):5.1f}%)  "
          f"2MiB-aligned start {m['a'] % (2 << 20) == 0}  {m['name']}  {numa.get(m['a'], '')}")
