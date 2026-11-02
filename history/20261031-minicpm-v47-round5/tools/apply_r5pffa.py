#!/usr/bin/env python3
"""Round-5 patch 20: IK_R5_PF also prefetches each thread's K/V positions of the next decode flash attention.

The r5 decode FA kernel (IK_R5_FA=1) gives thread ith the contiguous positions [ith*per, ith*per + per) of the
effective (unmasked) length, per = ceil(n_eff/nth) rounded up to 8, and reads each position's K and V row over all
KV heads. Its first DRAM fetch is the known slow part (FA-BOUNDARY.md). The planner now queues those rows from the
node-local KV copy in 32-position pieces, K and V alternating, so a budget cut still covers both. The mask is a graph
input, so n_eff is known before the node runs. Opt-in: IK_R5_PF_FA=1 (measured neutral, runs/79).
"""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'ggml/src/ggml.c'
s = p.read_text()
assert 'IK_R5_PF_FA' not in s

old = '''static int    r5pf_mode = -1, r5pf_lines = 4, r5pf_win = 32;
'''
assert s.count(old) == 1
s = s.replace(old, '''static int    r5pf_mode = -1, r5pf_lines = 4, r5pf_win = 32, r5pf_fa = 0;
''')
old = '''    r5pf_lines = lines; r5pf_win = win; r5pf_budget = budget;
'''
assert s.count(old) == 1
s = s.replace(old, old + '''    // K/V prefetch needs the r5 decode FA kernel's position split (IK_R5_FA=1)
    r5pf_fa = (e = getenv("IK_R5_FA")) && atoi(e) > 0 && (e = getenv("IK_R5_PF_FA")) && atoi(e) > 0;
''')

old = '''        const struct ggml_tensor * d = cg->nodes[j];
        if (d->op != GGML_OP_MUL_MAT || d->type != GGML_TYPE_F32) continue;
'''
assert s.count(old) == 1
s = s.replace(old, '''        const struct ggml_tensor * d = cg->nodes[j];
        if (d->op == GGML_OP_FLASH_ATTN_EXT && r5pf_fa) {
            // same eligibility and position split as r5fa::run
            const struct ggml_tensor * q = d->src[0], * k = d->src[1], * v = d->src[2], * m = d->src[3];
            if (!m || !m->data || d->src[4] || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 ||
                m->type != GGML_TYPE_F16 || q->ne[0] != 256 || q->ne[1] != 1 || q->ne[3] != 1 || k->ne[3] != 1) continue;
            if (k->nb[2] >= k->nb[1] || v->nb[2] >= v->nb[1]) continue;   // heads must lie inside a position row
            const uint16_t * mk = (const uint16_t *)m->data;
            int n_eff = (int)k->ne[1];
            while (n_eff > 0 && mk[n_eff - 1] == 0xFC00) --n_eff;
            const int per = ((n_eff + nth - 1)/nth + 7) & ~7;
            const int j0 = ith*per < n_eff ? ith*per : n_eff, j1 = j0 + per < n_eff ? j0 + per : n_eff;
            const size_t span_k = (size_t)(k->ne[2] - 1)*k->nb[2] + 2*256, span_v = (size_t)(v->ne[2] - 1)*v->nb[2] + 2*256;
            const char * kb = (const char *)ggml_numa_tensor_data(k, node), * vb = (const char *)ggml_numa_tensor_data(v, node);
            for (int jb = j0; jb < j1 && r5pf_t + 2 <= R5PF_MAXR && queued < r5pf_budget; jb += 32) {
                const int nj = j1 - jb < 32 ? j1 - jb : 32;
                const size_t nk = (size_t)(nj - 1)*k->nb[1] + span_k, nv = (size_t)(nj - 1)*v->nb[1] + span_v;
                r5pf_push(kb + (size_t)jb*k->nb[1], nk, j);
                r5pf_push(vb + (size_t)jb*v->nb[1], nv, j);
                queued += nk + nv;
            }
            continue;
        }
        if (d->op != GGML_OP_MUL_MAT || d->type != GGML_TYPE_F32) continue;
''')
p.write_text(s)
print('patched')
