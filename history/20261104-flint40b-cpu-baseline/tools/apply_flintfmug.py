#!/usr/bin/env python3
"""Flint patch 3: IK_R5_FMOE=1 IK_R5_FMOE_UG=1 also fuses Flint's MoE layer.

Flint has no shared-expert gate: routed up/gate, routed down, MUL_MULTI_ADD, ADD residual, shared up/gate, shared down,
ADD (7 nodes, against 9 on MiniCPM, which adds a gate MUL_MAT and a FUSED_MUL_UNARY). r5_fmug runs both patterns, on a
canonical 9-entry array with gt and gated NULL for Flint. Without a gate the tail is (routed sum + residual) + shared
down, the same additions in the same order as the unfused nodes, so results should be bitwise identical.
The socket-local variant (IK_R5_FMOE_LOCAL) stays MiniCPM only.
"""
import sys, pathlib
root = pathlib.Path(sys.argv[1])
p = root / 'ggml/src/ggml.c'
s = p.read_text()
assert 'r5_fmug2_match' not in s


def sub(old, new):
    global s
    assert s.count(old) == 1, (old[:80], s.count(old))
    s = s.replace(old, new)


# ---- matcher
sub('''// router prefix (12 nodes from the FFN norm to the routed up/gate); the work buffer also holds the router logits
''', '''// Flint's 7-node MoE layer: as r5_fmoe_match + r5_fmug_match without the shared gate
static bool r5_fmug2_match(const struct ggml_cgraph * g, int i, size_t wsize) {
    if (i + 6 >= g->n_nodes) R5F_FAIL;
    struct ggml_tensor * const * n = g->nodes + i;
    struct ggml_tensor * ugr = n[0], * down = n[1], * mma = n[2], * a1 = n[3], * ug = n[4], * sd = n[5], * out = n[6];
    if (down->op != GGML_OP_MUL_MAT_ID || mma->op != GGML_OP_MUL_MULTI_ADD || a1->op != GGML_OP_ADD ||
        ug->op != GGML_OP_FUSED_UP_GATE || sd->op != GGML_OP_MUL_MAT || out->op != GGML_OP_ADD) R5F_FAIL;
    const struct ggml_tensor * w0 = down->src[0], * par = down->src[1], * ids = down->src[2];
    const int64_t n_embd = down->ne[0];
    if (down->type != GGML_TYPE_F32 || down->ne[2] != 1 || down->ne[3] != 1 || w0->ne[3] != 1 || w0->ne[1] != n_embd) R5F_FAIL;
    if (ids->type != GGML_TYPE_I32 || ids->ne[1] != 1 || ids->ne[0] > 16 || down->ne[1] != ids->ne[0]) R5F_FAIL;
    if (par->type != GGML_TYPE_F32 || !ggml_is_contiguous(par) || par->ne[0] != w0->ne[0] || par->ne[2] != 1) R5F_FAIL;
    const struct ggml_tensor * W = mma->src[1];
    if (mma->src[0] != down || mma->src[2] || mma->src[3] || W->type != GGML_TYPE_F32 || W->ne[1] != ids->ne[0]) R5F_FAIL;
    const struct ggml_tensor * inp = r5_fmoe_inp(a1, mma);
    if (!inp || inp->type != GGML_TYPE_F32 || !ggml_is_contiguous(inp) || ggml_nelements(inp) != n_embd) R5F_FAIL;
    const struct ggml_tensor * ws = sd->src[0];
    if (sd->src[1] != ug || ug->type != GGML_TYPE_F32 || !ggml_is_contiguous(ug) || ug->ne[1] != 1 || ug->ne[2] != 1) R5F_FAIL;
    if (!ug->src[2] || ug->src[2]->type != GGML_TYPE_F32) R5F_FAIL;
    if (ws->ne[0] != ug->ne[0] || ws->ne[1] != n_embd || ws->ne[2] != 1 || ws->ne[3] != 1 || sd->type != GGML_TYPE_F32) R5F_FAIL;
    if (!((out->src[0] == a1 && out->src[1] == sd) || (out->src[1] == a1 && out->src[0] == sd))) R5F_FAIL;
    if (out->type != GGML_TYPE_F32 || !ggml_is_contiguous(out) || ggml_nelements(out) != n_embd || n_embd % 16) R5F_FAIL;
    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        if (r5_in(ug->src[s], n, 4)) R5F_FAIL;
    }
    const struct ggml_tensor * x = ugr->src[2], * up = ugr->src[0], * gate = ugr->src[1], * xs = ug->src[2];
    if (ugr->type != GGML_TYPE_F32 || !ggml_is_contiguous(ugr) || par != ugr || ugr->src[3] != ids) R5F_FAIL;
    if (!gate || gate->type != up->type || !ggml_are_same_shape(up, gate) || ugr->src[4] || ugr->src[5]) R5F_FAIL;
    if (!(x == xs || (x->view_src == xs && x->data == xs->data))) R5F_FAIL;
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || ggml_nrows(x) != 1 || ggml_nelements(x) != ggml_nelements(xs)) R5F_FAIL;
    if (up->ne[0] != x->ne[0] || up->ne[3] != 1 || ugr->ne[0] != up->ne[1] || ugr->ne[1] != ids->ne[0] || ugr->ne[2] != 1) R5F_FAIL;
    if (ug->src[1]->type != ug->src[0]->type || type_traits[ug->src[0]->type].vec_dot_type != type_traits[up->type].vec_dot_type) R5F_FAIL;
    if (ug->src[0]->ne[0] != x->ne[0] || ug->ne[0] != ug->src[0]->ne[1]) R5F_FAIL;
    const int cr = r5_fmug_rows();
    if (ugr->ne[0] % cr || ug->ne[0] % cr) R5F_FAIL;
    struct ggml_tensor * c[9] = { ugr, down, mma, a1, NULL, ug, sd, NULL, out };
    const struct r5_fmug_layout l = r5_fmug_lay(c);
    if (l.vt_u == GGML_TYPE_F32 || l.vt_r == GGML_TYPE_F32 || l.vt_s == GGML_TYPE_F32) R5F_FAIL;
    if (l.need > wsize) R5F_FAIL;
    return true;
}

// router prefix (12 nodes from the FFN norm to the routed up/gate); the work buffer also holds the router logits
''')

# ---- executor
sub('''    struct ggml_tensor * const * n = g->nodes + i;
    struct ggml_tensor * ugr = n[0], * down = n[1], * mma = n[2], * a1 = n[3], * gt = n[4], * ug = n[5], * sd = n[6], * gated = n[7], * out = n[8];
    const int ith = params->ith, nth = params->nth;

    if (ir < 0 && r5_local_mode()) {
''', '''    // Flint: 7 nodes, no shared gate (gt and gated stay NULL, gate value 1)
    struct ggml_tensor * const * g0 = g->nodes + i;
    const int len = g0[4]->op == GGML_OP_FUSED_UP_GATE ? 7 : 9;
    struct ggml_tensor * n[9] = { g0[0], g0[1], g0[2], g0[3], NULL, g0[4], g0[5], NULL, g0[6] };
    if (len == 9) memcpy(n, g0, sizeof(n));
    struct ggml_tensor * ugr = n[0], * down = n[1], * mma = n[2], * a1 = n[3], * gt = n[4], * ug = n[5], * sd = n[6], * gated = n[7], * out = n[8];
    const int ith = params->ith, nth = params->nth;

    if (ir < 0 && len == 9 && r5_local_mode()) {
''')
sub('''            struct ggml_tensor * const * sk = ir >= 0 ? g->nodes + ir : n;
            const int n_sk = ir >= 0 ? i - ir + 8 : 8;
            for (int k = i + 9; k < g->n_nodes; ++k) {
''', '''            struct ggml_tensor * const * sk = ir >= 0 ? g->nodes + ir : g0;
            const int n_sk = ir >= 0 ? i - ir + len - 1 : len - 1;
            for (int k = i + len; k < g->n_nodes; ++k) {
''')
sub('''    const struct ggml_tensor * up = ugr->src[0], * gate = ugr->src[1], * wg = gt->src[0], * w0 = down->src[0], * ws = sd->src[0];
    const struct r5_fmug_layout l = r5_fmug_lay(n);
    char * wq = (char *)params->wdata;
    const int n_ids = (int)ids->ne[0];
    const int64_t n_ff = ugr->ne[0], n_sh = ug->ne[0], n_embd = down->ne[0];
    const int numa_node = ggml_numa_node_for_thread(ith, nth);
    const int units = (int)(n_embd/16);
    const int u0 = (int)((int64_t)ith*units/nth), u1 = (int)((int64_t)(ith + 1)*units/nth);
    const int64_t r0 = 16*(int64_t)u0, nr = 16*(int64_t)(u1 - u0);

    // P0: quantize the FFN input, shared gate logit,''', '''    const struct ggml_tensor * up = ugr->src[0], * gate = ugr->src[1], * wg = gt ? gt->src[0] : NULL, * w0 = down->src[0], * ws = sd->src[0];
    const struct r5_fmug_layout l = r5_fmug_lay(n);
    char * wq = (char *)params->wdata;
    const int n_ids = (int)ids->ne[0];
    const int64_t n_ff = ugr->ne[0], n_sh = ug->ne[0], n_embd = down->ne[0];
    const int numa_node = ggml_numa_node_for_thread(ith, nth);
    const int units = (int)(n_embd/16);
    const int u0 = (int)((int64_t)ith*units/nth), u1 = (int)((int64_t)(ith + 1)*units/nth);
    const int64_t r0 = 16*(int64_t)u0, nr = 16*(int64_t)(u1 - u0);

    // P0: quantize the FFN input, shared gate logit,''')
sub('''    if (ith == 0) type_traits[l.vt_u].from_float(xd, wq, x->ne[0]);
    if (ith == nth - 1) {
        if (!iqk_mul_mat_4d(wg->ne[1], 1, wg->ne[0], wg->ne[2], wg->ne[3], x->ne[2], x->ne[3],
''', '''    if (ith == 0) type_traits[l.vt_u].from_float(xd, wq, x->ne[0]);
    if (gt && ith == nth - 1) {
        if (!iqk_mul_mat_4d(wg->ne[1], 1, wg->ne[0], wg->ne[2], wg->ne[3], x->ne[2], x->ne[3],
''')
sub('''    if (ir < 0) r5pf_moe(i + 8, w0, ws, sel, n_ids, numa_node, r0, nr);
''', '''    if (ir < 0) r5pf_moe(i + len - 1, w0, ws, sel, n_ids, numa_node, r0, nr);
''')
sub('''        const float xg = *(const float *)(wq + l.off_g);
        const float gv = gated->op_params[0] == GGML_UNARY_OP_SILU ? ggml_silu_f32(xg) : 1.0f/(1.0f + expf(-xg));
        const float limit = *(const float *)(gated->op_params + 1);
        iqk_r5_moe_tail((int)nr, n_ids, buf, nr, wts, res, sh, gv, limit, (float *)out->data + r0, sh + nr, sh + 2*nr);
    }
    return i + 8;
}
''', '''        const float xg = *(const float *)(wq + l.off_g);
        const float gv = !gated ? 1.0f : gated->op_params[0] == GGML_UNARY_OP_SILU ? ggml_silu_f32(xg) : 1.0f/(1.0f + expf(-xg));
        const float limit = gated ? *(const float *)(gated->op_params + 1) : 0.0f;
        iqk_r5_moe_tail((int)nr, n_ids, buf, nr, wts, res, sh, gv, limit, (float *)out->data + r0, sh + nr, sh + 2*nr);
    }
    return i + len - 1;
}
''')

# ---- dispatch
sub('''                if (r5_fmoe_mode() > 0 && r5_fmug_mode() > 0 && r5_fmug_match(cgraph, i, params->wsize)) {
                    i = r5_fmug(params, cgraph, i, -1);
''', '''                if (r5_fmoe_mode() > 0 && r5_fmug_mode() > 0 &&
                    (r5_fmug_match(cgraph, i, params->wsize) || r5_fmug2_match(cgraph, i, params->wsize))) {
                    i = r5_fmug(params, cgraph, i, -1);
''')
p.write_text(s)
print('patched')
