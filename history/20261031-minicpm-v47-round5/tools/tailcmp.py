#!/usr/bin/env python3
"""Per-graph cost of the MoE tail (routed down .. l_out) in r5prof dumps: tailcmp.py PROF [...]

Sums wall / maxc / meanc over the nodes named ffn_inp_normed .. l_out (the MoE block: router, up/gate, down, combine;
fused: one span-8 node per layer named ffn_moe_down) plus the per-graph total, and prints layer 3.
"""
import re, sys

TAIL = ('ffn_inp_normed', 'ffn_moe_logits', 'ffn_moe_probs', 'ffn_moe_weights_sum', 'ffn_moe_gate_par', 'ffn_moe_down', 'routed_out', 'routed_out_with_inp', 'shared_expert_gate', 'ffn_up_gate',
        'ffn_shexp_out', 'ffn_shexp_gated', 'l_out')

for path in sys.argv[1:]:
    total = None; rows = []; sec = False
    for line in open(path):
        if line.startswith('# per graph'):
            total = float(re.search(r'total=([\d.]+)', line).group(1))
        elif line.startswith('## by node'):
            sec = True
        elif sec and line[:5].strip().isdigit():
            f = line.split()
            nums = list(map(float, f[-6:]))
            rows.append((int(f[0]), int(f[1]), f[2], f[3], nums[0], nums[1], nums[2]))
    tail = [r for r in rows if r[3].rsplit('-', 1)[0] in TAIL]
    w = sum(r[4] for r in tail); x = sum(r[5] for r in tail); m = sum(r[6] for r in tail)
    print(f'== {path}\n   graph {total/1e3:.2f} ms   MoE tail: {len(tail)} nodes  wall {w/1e3:.3f} ms  maxc {x/1e3:.3f}  meanc {m/1e3:.3f}')
    for r in tail:
        if r[3].endswith('-3'):
            print(f'     {r[0]:5d} {r[1]:2d} {r[2]:17s} {r[3]:26s} wall {r[4]:7.2f} maxc {r[5]:7.2f} meanc {r[6]:7.2f}')
