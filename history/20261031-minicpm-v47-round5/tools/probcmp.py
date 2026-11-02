#!/usr/bin/env python3
"""Compare per-token top-5 probabilities of r5bench --probs runs against a reference run.

  probcmp.py REF_RUN_JSON CAND_RUN_JSON [...]
For each candidate: first token index where the greedy token differs, and up to that point the
max |p_ref - p_cand| of the chosen token, plus the top-1/top-2 margin of the reference at the
divergence point (a small margin means a near-tie, i.e. reduction-order sensitivity).
"""
import json, sys

def toks(run):
    out = []
    for step in run['completion_probabilities']:
        probs = step.get('probs') or step.get('top_probs') or []
        cand = [(p.get('tok_str', p.get('token')), p.get('prob', p.get('probability', 0.0))) for p in probs]
        chosen = step.get('content', step.get('token'))
        out.append((chosen, cand))
    return out

ref = json.load(open(sys.argv[1]))
R = toks(ref)
print('reference', ref['tag'])
for path in sys.argv[2:]:
    c = json.load(open(path))
    C = toks(c)
    div, maxd, maxd_at = None, 0.0, -1
    for i, ((rt, rp), (ct, cp)) in enumerate(zip(R, C)):
        if rt != ct:
            div = i
            break
        rd = dict(rp); cd = dict(cp)
        for t, p in rp:
            if t in cd:
                d = abs(p - cd[t])
                if d > maxd:
                    maxd, maxd_at = d, i
    line = f"{c['tag']:28s} "
    if div is None:
        line += f'all {min(len(R), len(C))} tokens identical; '
    else:
        rp = R[div][1]
        margin = rp[0][1] - rp[1][1] if len(rp) > 1 else float('nan')
        line += (f'diverges at token {div} (ref {R[div][0]!r} vs {C[div][0]!r}; ref top1-top2 margin {margin:.4f}; '
                 f'ref top2 {[(t, round(p, 4)) for t, p in rp[:2]]}; cand top2 {[(t, round(p, 4)) for t, p in C[div][1][:2]]}); ')
    line += f'max |dp| before divergence {maxd:.4f} at token {maxd_at}'
    print(line)
