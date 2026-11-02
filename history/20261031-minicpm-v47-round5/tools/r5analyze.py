#!/usr/bin/env python3
"""Summarise an r5prof dump: where does per-token graph time go?

wall = meanc (useful compute) + (maxc - meanc) (imbalance) + rest (barrier tail,
wake/start latency).
"""
import sys, re

def load(path):
    nodes, hdr = [], {}
    sec = None
    for line in open(path):
        if line.startswith('# r5prof'):
            hdr.update(dict(kv.split('=') for kv in line.split()[2:]))
        elif line.startswith('# per graph'):
            hdr['graph'] = line.strip()
        elif line.startswith('## by node'):
            sec = 'node'
        elif sec == 'node' and line[:5].strip().isdigit():
            f = line.split()
            # idx span op name type0 ne0 ne1 wall maxc meanc tail lat latmax ; name may contain spaces
            nums = list(map(float, f[-6:]))
            idx, span, op = int(f[0]), int(f[1]), f[2]
            mid = f[3:-6]
            type0, ne0, ne1 = mid[-3], mid[-2], mid[-1]
            name = ' '.join(mid[:-3])
            nodes.append(dict(idx=idx, span=span, op=op, name=name, type0=type0, ne0=ne0, ne1=ne1,
                              wall=nums[0], maxc=nums[1], meanc=nums[2], tail=nums[3], lat=nums[4], latmax=nums[5]))
    return hdr, nodes

def main():
    for path in sys.argv[1:]:
        hdr, nodes = load(path)
        W = sum(n['wall'] for n in nodes)
        mean = sum(n['meanc'] for n in nodes)
        imb = sum(n['maxc'] - n['meanc'] for n in nodes)
        rest = W - mean - imb
        print(f'== {path}')
        print(f"   {hdr.get('graph')}")
        print(f'   executed nodes {len(nodes)}  wall {W/1e3:.2f} ms  mean-compute {mean/1e3:.2f}  imbalance {imb/1e3:.2f}  sync/latency {rest/1e3:.2f}')
        big = [n for n in nodes if n['wall'] >= 20]
        small = [n for n in nodes if n['wall'] < 20]
        for lab, grp in (('nodes >= 20us', big), ('nodes < 20us', small)):
            w = sum(n['wall'] for n in grp); m = sum(n['meanc'] for n in grp); x = sum(n['maxc'] for n in grp)
            print(f'   {lab:14s} n={len(grp):5d} wall {w/1e3:6.2f} ms  meanc {m/1e3:6.2f}  maxc {x/1e3:6.2f}  sync {(w-x)/1e3:6.2f}')
        st = [n for n in nodes if n['meanc'] < 0.25*n['maxc'] and n['maxc'] > 1]
        print(f'   effectively single-threaded nodes (meanc < maxc/4): {len(st)}  wall {sum(n["wall"] for n in st)/1e3:.2f} ms')
        fused = [n for n in nodes if n['span'] > 1]
        print(f'   fused nodes (span>1): {len(fused)}')
        ops = {}
        for n in fused:
            ops.setdefault((n['op'], re.sub(r'-\d+', '-N', n['name']), n['span']), 0)
            ops[(n['op'], re.sub(r'-\d+', '-N', n['name']), n['span'])] += 1
        for k, v in sorted(ops.items(), key=lambda kv: -kv[1])[:12]:
            print(f'      {v:3d} x {k}')
        # one representative layer of each kind
        print('   first recurrent layer + first attention layer nodes:')
        for n in nodes:
            if re.search(r'-(0|3)$', n['name']) or re.search(r'-(0|3) ', n['name']):
                print(f"     {n['idx']:5d} {n['span']:2d} {n['op']:18s} {n['name'][:34]:34s} {n['type0']:5s} {n['ne0']:>20s} {n['ne1']:>18s} wall {n['wall']:7.2f} maxc {n['maxc']:7.2f} meanc {n['meanc']:7.2f} tail {n['tail']:5.2f} lat {n['lat']:5.2f}")

main()
