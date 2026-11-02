#!/usr/bin/env python3
"""Round-5 patch 16: fold the MoE router into the fused MoE node (IK_R5_FMOE_UG=2).

Decode graph per MoE layer before the routed up/gate (12 nodes, 4 barriers):
  FUSED_RMS_NORM ffn_inp_normed | MUL_MAT ffn_moe_logits | SOFT_MAX + RESHAPE + ARGSORT + VIEW + GET_ROWS (executor-fused
  iqk_topk_moe) | RESHAPE | SUM_ROWS + DIV (executor-fused iqk_sumrows_div) | RESHAPE weights | RESHAPE normed input
With IK_R5_FMOE_UG=2 the match starts at the norm and r5_fmug runs all 21 nodes up to l_out:
  R0  every thread computes the normed input itself (ggml_compute_forward_fused_rms_norm with ith=0, nth=1 on a private
      copy, so the same bits); thread 0 quantizes it, the last thread computes the shared gate logit, all threads
      split the router logits rows (same iqk_mul_mat_4d call as the MUL_MAT node) into the work buffer     | barrier
  R1  every thread runs iqk_topk_moe and iqk_sumrows_div on private copies (the functions the executor's fused
      SOFT_MAX and SUM_ROWS paths call), giving the expert ids and normalized weights without another barrier
  P1..P3 as in patch 15.
Barriers from the norm to l_out: 8 -> 4 (including the executor's). The MoE expert prefetch hook reads the graph's
ids, so the router version is only used when that prefetch is off.
"""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'ggml/src/ggml.c'
s = p.read_text()
assert 'r5_fmrt' not in s

def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:80])
    s = s.replace(old, new)

# ---- router match, placed after r5_fmug_match ----
rep('''#undef R5F_FAIL

static int r5_fmug(struct ggml_compute_params * params, const struct ggml_cgraph * g, int i) {''',
r'''// router prefix (12 nodes from the FFN norm to the routed up/gate); the work buffer also holds the router logits
static bool r5_fmrt_match(const struct ggml_cgraph * g, int i, size_t wsize) {
    if (i + 20 >= g->n_nodes) R5F_FAIL;
    struct ggml_tensor * const * n = g->nodes + i;
    const struct ggml_tensor * nrm = n[0], * lg = n[1], * sm = n[2], * srt = n[4], * tk = n[5], * gr = n[6], * w8 = n[7];
    const struct ggml_tensor * sr = n[8], * dv = n[9], * wn = n[10], * xr = n[11];
    if (lg->op != GGML_OP_MUL_MAT || sm->op != GGML_OP_SOFT_MAX || n[3]->op != GGML_OP_RESHAPE || srt->op != GGML_OP_ARGSORT ||
        tk->op != GGML_OP_VIEW || gr->op != GGML_OP_GET_ROWS || w8->op != GGML_OP_RESHAPE || sr->op != GGML_OP_SUM_ROWS ||
        dv->op != GGML_OP_DIV || wn->op != GGML_OP_RESHAPE || xr->op != GGML_OP_RESHAPE || n[12]->op != GGML_OP_MOE_FUSED_UP_GATE) R5F_FAIL;
    if (!r5_fmug_match(g, i + 12, SIZE_MAX)) return false;   // nodes 12..20 are the r5_fmug pattern
    const struct ggml_tensor * ugr = n[12], * down = n[13], * mma = n[14], * ug = n[17];
    if (nrm->type != GGML_TYPE_F32 || !ggml_is_contiguous(nrm) || ggml_nrows(nrm) != 1 || !nrm->src[1]) R5F_FAIL;
    if (lg->src[1] != nrm || lg->type != GGML_TYPE_F32 || lg->src[0]->type != GGML_TYPE_F32 || !ggml_is_contiguous(lg) || ggml_nrows(lg) != 1) R5F_FAIL;
    if (lg->src[0]->ne[0] != nrm->ne[0] || lg->ne[0] > 1024 || ggml_nrows(lg->src[0]) != lg->ne[0]) R5F_FAIL;
    // the conditions under which the executor fuses SOFT_MAX .. GET_ROWS and SUM_ROWS + DIV
    if (sm->src[0] != lg || sm->type != GGML_TYPE_F32 || gr->type != GGML_TYPE_F32 || tk->type != GGML_TYPE_I32) R5F_FAIL;
    if (srt->src[0] != sm || tk->view_src != srt || gr->src[1] != tk || gr->ne[1] != tk->ne[0]) R5F_FAIL;
    if (w8->src[0] != gr || sr->src[0] != w8 || dv->src[0] != w8 || dv->src[1] != sr || wn->src[0] != dv) R5F_FAIL;
    if (!ggml_is_contiguous(w8) || ggml_nrows(w8) != 1 || !ggml_is_contiguous(dv) || dv->type != GGML_TYPE_F32) R5F_FAIL;
    // the fused pattern must consume exactly these ids, weights and normed input
    if (ugr->src[3] != tk || down->src[2] != tk || mma->src[1] != wn || ug->src[2] != nrm || ugr->src[2] != xr || xr->src[0] != nrm) R5F_FAIL;
    if (GGML_PAD(r5_fmug_lay(n + 12).need, 64) + lg->ne[0]*sizeof(float) > wsize) R5F_FAIL;
    return true;
}

#undef R5F_FAIL

// i: the routed up/gate node; ir: the FFN norm when the router prefix is fused too (IK_R5_FMOE_UG=2), else -1
static int r5_fmug(struct ggml_compute_params * params, const struct ggml_cgraph * g, int i, int ir) {''')

# ---- consumer check covers the router nodes too ----
rep('''            for (int k = i + 9; k < g->n_nodes; ++k) {
                for (int s = 0; s < GGML_MAX_SRC; ++s) {
                    if (r5_in(g->nodes[k]->src[s], n, 8)) {''',
'''            struct ggml_tensor * const * sk = ir >= 0 ? g->nodes + ir : n;
            const int n_sk = ir >= 0 ? i - ir + 8 : 8;
            for (int k = i + 9; k < g->n_nodes; ++k) {
                for (int s = 0; s < GGML_MAX_SRC; ++s) {
                    if (r5_in(g->nodes[k]->src[s], sk, n_sk)) {''')

# ---- P0 / router phase ----
rep('''    // P0: quantize the FFN input, shared gate logit, snapshot ids, weights and the residual slice
    if (ith == 0) type_traits[l.vt_u].from_float((const float *)x->data, wq, x->ne[0]);
    if (ith == nth - 1) {
        if (!iqk_mul_mat_4d(wg->ne[1], 1, wg->ne[0], wg->ne[2], wg->ne[3], x->ne[2], x->ne[3],
                            wg->nb[2], wg->nb[3], x->nb[2], x->nb[3], gt->nb[2]/sizeof(float), gt->nb[3]/sizeof(float),
                            wg->type, ggml_numa_tensor_data(wg, numa_node), wg->nb[1], x->type, x->data, x->nb[1],
                            (float *)(wq + l.off_g), gt->nb[1]/sizeof(float), 0, 1)) GGML_ABORT("r5_fmug: shared gate type not supported");
    }
    int32_t sel[16];
    float wts[16];
    for (int j = 0; j < n_ids; ++j) {
        sel[j] = *(const int32_t *)((const char *)ids->data + j*ids->nb[0]);
        wts[j] = *(const float *)((const char *)W->data + j*W->nb[1]);
    }
    float res[nr > 0 ? nr : 1];
    if (nr > 0) memcpy(res, (const float *)inp->data + r0, nr*sizeof(float));
    if (params->shared->cplan && params->shared->cplan->moe_expert_prefetch) {
        ggml_moe_prefetch_kernel_hook(ugr, ith);
    }
    if (ith == 0) {
        atomic_store(&params->shared->current_chunk, nth);
        r5_claim_reset(params->shared);
    }
    ggml_barrier(params->shared);
''', '''    // P0: quantize the FFN input, shared gate logit, snapshot ids, weights and the residual slice
    // (router version: every thread computes the normed input itself, and the router logits are split over the threads)
    float xn[ir >= 0 ? x->ne[0] : 1];
    const float * xd = (const float *)x->data;
    float * lgb = (float *)(wq + GGML_PAD(l.need, 64));
    if (ir >= 0) {
        struct ggml_tensor tn = *g->nodes[ir];
        tn.data = xn;
        struct ggml_compute_params p1 = *params;
        p1.ith = 0;
        p1.nth = 1;
        ggml_compute_forward_fused_rms_norm(&p1, &tn);
        xd = xn;
    }
    if (ith == 0) type_traits[l.vt_u].from_float(xd, wq, x->ne[0]);
    if (ith == nth - 1) {
        if (!iqk_mul_mat_4d(wg->ne[1], 1, wg->ne[0], wg->ne[2], wg->ne[3], x->ne[2], x->ne[3],
                            wg->nb[2], wg->nb[3], x->nb[2], x->nb[3], gt->nb[2]/sizeof(float), gt->nb[3]/sizeof(float),
                            wg->type, ggml_numa_tensor_data(wg, numa_node), wg->nb[1], x->type, xd, x->nb[1],
                            (float *)(wq + l.off_g), gt->nb[1]/sizeof(float), 0, 1)) GGML_ABORT("r5_fmug: shared gate type not supported");
    }
    int32_t sel[16];
    float wts[16];
    if (ir >= 0) {
        const struct ggml_tensor * lg = g->nodes[ir + 1], * wl = lg->src[0], * xl = lg->src[1];
        if (!iqk_mul_mat_4d(wl->ne[1], 1, wl->ne[0], wl->ne[2], wl->ne[3], xl->ne[2], xl->ne[3],
                            wl->nb[2], wl->nb[3], xl->nb[2], xl->nb[3], lg->nb[2]/sizeof(float), lg->nb[3]/sizeof(float),
                            wl->type, ggml_numa_tensor_data(wl, numa_node), wl->nb[1], xl->type, xd, xl->nb[1],
                            lgb, lg->nb[1]/sizeof(float), ith, nth)) GGML_ABORT("r5_fmug: router logits type not supported");
    } else {
        for (int j = 0; j < n_ids; ++j) {
            sel[j] = *(const int32_t *)((const char *)ids->data + j*ids->nb[0]);
            wts[j] = *(const float *)((const char *)W->data + j*W->nb[1]);
        }
    }
    float res[nr > 0 ? nr : 1];
    if (nr > 0) memcpy(res, (const float *)inp->data + r0, nr*sizeof(float));
    if (ir < 0 && params->shared->cplan && params->shared->cplan->moe_expert_prefetch) {
        ggml_moe_prefetch_kernel_hook(ugr, ith);
    }
    if (ith == 0) {
        atomic_store(&params->shared->current_chunk, nth);
        r5_claim_reset(params->shared);
    }
    ggml_barrier(params->shared);

    if (ir >= 0) {
        // router: expert ids and normalized weights, computed by every thread with the functions the executor's fused
        // SOFT_MAX .. GET_ROWS and SUM_ROWS + DIV paths call, on private copies
        const int n_exp = (int)g->nodes[ir + 1]->ne[0];
        int32_t ids_all[n_exp];
        float w_raw[16];
        iqk_topk_moe(n_exp, n_ids, 1, lgb, w_raw, ids_all, 0, 1);
        struct ggml_tensor ts = *g->nodes[ir + 7], td = *g->nodes[ir + 9];
        ts.data = w_raw;
        td.data = wts;
        td.src[0] = &ts;
        iqk_sumrows_div(&td, 0, 1);
        for (int j = 0; j < n_ids; ++j) sel[j] = ids_all[j];
    }
''')

# ---- dispatch ----
rep('                    i = r5_fmug(params, cgraph, i);\n', '                    i = r5_fmug(params, cgraph, i, -1);\n')
rep('''        case GGML_OP_FUSED_RMS_NORM:
            {
                ggml_compute_forward_fused_rms_norm(params, tensor);
            } break;
''', '''        case GGML_OP_FUSED_RMS_NORM:
            {
                if (r5_fmoe_mode() > 0 && r5_fmug_mode() > 1 && !(params->shared->cplan && params->shared->cplan->moe_expert_prefetch) &&
                    r5_fmrt_match(cgraph, i, params->wsize)) {
                    i = r5_fmug(params, cgraph, i + 12, i);
                } else {
                    if (r5_fmoe_mode() == 3 && r5_fmug_mode() > 1 && params->ith == 0 && i + 2 < cgraph->n_nodes &&
                        cgraph->nodes[i + 2]->op == GGML_OP_SOFT_MAX && ggml_nrows(tensor) == 1) {
                        static int r5r_m = 0;
                        if (r5r_m++ < 100) fprintf(stderr, "r5_fmrt: no match at node %d %s (check line %d)\\n", i, tensor->name, r5f_line);
                    }
                    ggml_compute_forward_fused_rms_norm(params, tensor);
                }
            } break;
''')
p.write_text(s)
print('patched')
