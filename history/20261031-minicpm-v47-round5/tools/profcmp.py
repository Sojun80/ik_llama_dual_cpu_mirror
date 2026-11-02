#!/usr/bin/env python3
# Compare two r5prof decode profiles: per-class wall / compute / idle (wall - mean thread compute), per graph
import sys, re

def load(path):
    head, cls, thr = {}, {}, []
    sec = None
    for line in open(path):
        if line.startswith('# per graph'):
            head = {k: float(v) for k, v in re.findall(r'(\w+)(?:\([^)]*\))?=([\d.]+)', line)}
        if line.startswith('## '):
            sec = line[3:10]
            continue
        if sec == 'by clas' and '|' in line and not line.startswith('op|'):
            f = line.split()
            # name may contain spaces; numbers are the 8 fields before type0
            name = ' '.join(f[:-9])
            n, wall, maxc, meanc, tail, lat = (float(x) for x in f[-9:-3])
            key = re.sub(r'node_\d+', 'node_N', name)
            key = re.sub(r'_l\d+', '_lN', key)
            a = cls.setdefault(key, [0.0]*6)
            for k, v in enumerate((n, wall, maxc, meanc, tail, lat)):
                a[k] += v
        if sec == 'per thr' and line.startswith('thr'):
            f = line.split()
            thr.append(float(f[-1]))
    return head, cls, thr

a, b = sys.argv[1], sys.argv[2]
ha, ca, ta = load(a)
hb, cb, tb = load(b)
tot_a, tot_b = ha['total'], hb['total']
print(f'per token: {tot_a:.1f} -> {tot_b:.1f} us')
ia = tot_a - sum(ta)/len(ta)
ib = tot_b - sum(tb)/len(tb)
print(f'mean thread compute: {sum(ta)/len(ta):.1f} -> {sum(tb)/len(tb):.1f} us; mean idle {ia:.1f} -> {ib:.1f} us')
print(f'thread compute min/max: {min(ta):.0f}/{max(ta):.0f} -> {min(tb):.0f}/{max(tb):.0f}')
print()
print(f'{"class":58s} {"n":>4s} {"wallA":>8s} {"wallB":>8s} {"meanA":>8s} {"meanB":>8s} {"idleA":>7s} {"idleB":>7s} {"maxB-meanB":>10s} {"latB":>6s}')
rows = []
for k in set(ca) | set(cb):
    x, y = ca.get(k, [0]*6), cb.get(k, [0]*6)
    rows.append((k, x, y))
rows.sort(key=lambda r: -(r[2][1] - r[2][3]))
sa = sb = 0.0
for k, x, y in rows:
    sa += x[1] - x[3]; sb += y[1] - y[3]
    if y[1] < 40 and x[1] < 40: continue
    print(f'{k[:58]:58s} {y[0]:4.0f} {x[1]:8.1f} {y[1]:8.1f} {x[3]:8.1f} {y[3]:8.1f} {x[1]-x[3]:7.1f} {y[1]-y[3]:7.1f} {y[2]-y[3]:10.1f} {y[5]:6.1f}')
print(f'{"sum of class idle":58s} {"":4s} {"":8s} {"":8s} {"":8s} {"":8s} {sa:7.1f} {sb:7.1f}')
