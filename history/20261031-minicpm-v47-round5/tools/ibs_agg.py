#!/usr/bin/env python3
"""Aggregate IBS op load samples from `perf script -F ip,sym,addr,weight,data_src` (stdin).

Line layout: ADDR DSRC_HEX |OP x|LVL y|SNP..|TLB z|LCK..|BLK  N/A  WEIGHT IP SYM
Per symbol (top N by load samples): share and mean latency (cycles) per data-source level,
plus for one focus symbol the hottest IPs among RAM-hit loads.
"""
import re, sys, collections
topn = int(sys.argv[1]) if len(sys.argv) > 1 else 8
focus = sys.argv[2] if len(sys.argv) > 2 else None
pat = re.compile(r'^\s*([0-9a-f]+)\s+\S+\s+\|OP (\w+)\|LVL ([^|]*)\|SNP [^|]*\|TLB ([^|]*)\|.*?BLK\s+N/A\s+(\d+)\s+([0-9a-f]+)\s+(.*)$')
stats = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
ram_ips = collections.defaultdict(lambda: [0, 0])
tlb = collections.defaultdict(collections.Counter)
for line in sys.stdin:
    m = pat.match(line)
    if not m:
        continue
    addr, op, lvl, tl, w, ip, sym = m.groups()
    if op != 'LOAD':
        continue
    w = int(w)
    sym = sym.split('(')[0][-70:]
    s = stats[sym][lvl.strip()]
    s[0] += 1; s[1] += w
    tlb[sym][tl.strip()] += 1
    if focus and focus in sym and 'RAM' in lvl:
        r = ram_ips[ip]; r[0] += 1; r[1] += w
tot = {k: sum(v[0] for v in d.values()) for k, d in stats.items()}
for sym in sorted(tot, key=lambda k: -tot[k])[:topn]:
    lat = sum(v[1] for v in stats[sym].values())
    print(f'{sym}: {tot[sym]} loads, total latency {lat/1e6:.2f} Mcycles; TLB {dict(tlb[sym].most_common(3))}')
    for lvl, (n, w) in sorted(stats[sym].items(), key=lambda kv: -kv[1][1])[:5]:
        print(f'    {lvl:30s} {100*n/tot[sym]:6.2f}% of loads  mean {w/n:7.1f} cyc  share of latency {100*w/max(lat,1):5.1f}%')
if focus:
    print(f'hottest RAM-hit IPs in {focus}:')
    for ip, (n, w) in sorted(ram_ips.items(), key=lambda kv: -kv[1][1])[:10]:
        print(f'    {ip}  {n:6d} samples  mean {w/n:6.1f} cyc')
