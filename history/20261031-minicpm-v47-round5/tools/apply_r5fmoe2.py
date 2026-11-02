#!/usr/bin/env python3
"""Round-5 patch 14b: make the fused MoE tail safe against compute-buffer reuse.

The graph allocator places later nodes over buffers whose last consumer has run, so in the decode
graph the shared-expert gate/up outputs (nodes 3, 4) may sit on top of par, the routing weights,
ids or the residual input, which the original order had finished reading. The fused path now:
  P0  quantize par into the work buffer (above the region node 4 uses) and snapshot ids, weights
      and this thread's slice of the residual input onto the stack           | barrier
  P1  node 3 (gate)                                                          | barrier (IK_R5_FMOE_B=0 drops it)
  P2  node 4 (up/gate)                                                       | barrier
  P3  quantize node 4's output, snapshot the gate value                      | barrier
  P4  per-thread row slice: routed + shared down rows, combine, write out
After the P3 barrier no thread reads a graph tensor, so `out` may overlap anything. Also replaces
the earlier debug prints with one line per decode miss (IK_R5_FMOE=3: failing check line).
"""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'ggml/src/ggml.c'
s = p.read_text()

a = s.index('// ---- round-5: fused MoE tail for single-token decode')
end_marker = '''static int ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor,
        const struct ggml_cgraph * cgraph, int i) {
'''
b = s.index(end_marker, s.index('static int r5_fmoe(struct ggml_compute_params'))

NEW = r'''// ---- round-5: fused MoE tail for single-token decode (IK_R5_FMOE, see apply_r5fmoe.py / apply_r5fmoe2.py) ----
static int r5_fmoe_mode(void) {
    static int mode = -1;
    if (mode < 0) { const char * e = getenv("IK_R5_FMOE"); mode = e ? atoi(e) : 0; }
    return mode;
}

static inline bool r5_in(const struct ggml_tensor * t, struct ggml_tensor * const * n, int k) {
    for (int j = 0; j < k; ++j) if (t && (t == n[j] || t->view_src == n[j])) return true;
    return false;
}

static const struct ggml_tensor * r5_fmoe_inp(const struct ggml_tensor * a1, const struct ggml_tensor * mma) {
    return a1->src[0] == mma ? a1->src[1] : a1->src[1] == mma ? a1->src[0] : NULL;
}

// work-buffer layout: [0, off_r) node 4's quantized input | routed rows at off_r | shared row at off_s
struct r5_fmoe_layout { enum ggml_type vt_r, vt_s; size_t rs_r, rs_s, off_r, off_s, need; };

static struct r5_fmoe_layout r5_fmoe_lay(struct ggml_tensor * const * n) {
    const struct ggml_tensor * w0 = n[0]->src[0], * par = n[0]->src[1], * ug = n[4], * ws = n[5]->src[0];
    struct r5_fmoe_layout l;
    l.vt_r = type_traits[w0->type].vec_dot_type;
    l.vt_s = type_traits[ws->type].vec_dot_type;
    l.rs_r = ggml_row_size(l.vt_r, par->ne[0]);
    l.rs_s = ggml_row_size(l.vt_s, ug->ne[0]);
    const struct ggml_tensor * ug_in = ug->src[2];
    l.off_r = GGML_PAD(ggml_row_size(type_traits[ug->src[0]->type].vec_dot_type, ug_in->ne[0])*ggml_nrows(ug_in), 64);
    l.off_s = l.off_r + GGML_PAD(par->ne[1]*l.rs_r, 64);
    l.need  = l.off_s + l.rs_s;
    return l;
}

static int r5f_line;
#define R5F_FAIL do { r5f_line = __LINE__; return false; } while (0)

static bool r5_fmoe_match(const struct ggml_cgraph * g, int i, size_t wsize) {
    if (i + 7 >= g->n_nodes) R5F_FAIL;
    struct ggml_tensor * const * n = g->nodes + i;
    const struct ggml_tensor * down = n[0], * mma = n[1], * a1 = n[2], * gt = n[3], * ug = n[4], * sd = n[5], * gated = n[6], * out = n[7];
    if (mma->op != GGML_OP_MUL_MULTI_ADD || a1->op != GGML_OP_ADD || gt->op != GGML_OP_MUL_MAT || ug->op != GGML_OP_FUSED_UP_GATE ||
        sd->op != GGML_OP_MUL_MAT || gated->op != GGML_OP_FUSED_MUL_UNARY || out->op != GGML_OP_ADD) R5F_FAIL;
    const struct ggml_tensor * w0 = down->src[0], * par = down->src[1], * ids = down->src[2];
    const int64_t n_embd = down->ne[0];
    if (down->type != GGML_TYPE_F32 || down->ne[2] != 1 || down->ne[3] != 1 || w0->ne[3] != 1 || w0->ne[1] != n_embd) R5F_FAIL;
    if (ids->type != GGML_TYPE_I32 || ids->ne[1] != 1 || ids->ne[0] > 16 || down->ne[1] != ids->ne[0]) R5F_FAIL;
    if (par->type != GGML_TYPE_F32 || !ggml_is_contiguous(par) || par->ne[0] != w0->ne[0] || par->ne[2] != 1) R5F_FAIL;
    const struct ggml_tensor * W = mma->src[1];
    if (mma->src[0] != down || mma->src[2] || mma->src[3] || W->type != GGML_TYPE_F32 || W->ne[1] != ids->ne[0]) R5F_FAIL;
    const struct ggml_tensor * inp = r5_fmoe_inp(a1, mma);
    if (!inp || inp->type != GGML_TYPE_F32 || !ggml_is_contiguous(inp) || ggml_nelements(inp) != n_embd) R5F_FAIL;
    if (gated->src[0] != gt || gated->src[1] != sd) R5F_FAIL;
    if (gated->op_params[0] != GGML_UNARY_OP_SIGMOID && gated->op_params[0] != GGML_UNARY_OP_SILU) R5F_FAIL;
    if (gt->type != GGML_TYPE_F32 || ggml_nelements(gt) != 1 || gt->src[0]->type != GGML_TYPE_F32 || gt->src[1]->type != GGML_TYPE_F32) R5F_FAIL;
    const struct ggml_tensor * ws = sd->src[0];
    if (sd->src[1] != ug || ug->type != GGML_TYPE_F32 || !ggml_is_contiguous(ug) || ug->ne[1] != 1 || ug->ne[2] != 1) R5F_FAIL;
    if (!ug->src[2] || ug->src[2]->type != GGML_TYPE_F32) R5F_FAIL;
    if (ws->ne[0] != ug->ne[0] || ws->ne[1] != n_embd || ws->ne[2] != 1 || ws->ne[3] != 1 || sd->type != GGML_TYPE_F32) R5F_FAIL;
    if (!((out->src[0] == a1 && out->src[1] == gated) || (out->src[1] == a1 && out->src[0] == gated))) R5F_FAIL;
    if (out->type != GGML_TYPE_F32 || !ggml_is_contiguous(out) || ggml_nelements(out) != n_embd || n_embd % 16) R5F_FAIL;
    // nodes 3 and 4 run first, so they must not read the routed part
    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        if (r5_in(gt->src[s], n, 3) || r5_in(ug->src[s], n, 3)) R5F_FAIL;
    }
    const struct r5_fmoe_layout l = r5_fmoe_lay(n);
    if (l.vt_r == GGML_TYPE_F32 || l.vt_s == GGML_TYPE_F32 || l.need > wsize) R5F_FAIL;
    return true;
}

#undef R5F_FAIL

static int ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor,
        const struct ggml_cgraph * cgraph, int i);

static int r5_fmoe(struct ggml_compute_params * params, const struct ggml_cgraph * g, int i) {
    struct ggml_tensor * const * n = g->nodes + i;
    struct ggml_tensor * down = n[0], * mma = n[1], * a1 = n[2], * gt = n[3], * ug = n[4], * sd = n[5], * gated = n[6], * out = n[7];
    const int ith = params->ith, nth = params->nth;

    if (r5_fmoe_mode() == 2 && ith == 0) {
        // debug: the skipped nodes must have no consumers outside the pattern
        static int n_checked = 0, n_bad = 0;
        if (n_checked < 200) {
            struct ggml_tensor * const skipped[5] = { n[0], n[1], n[2], n[5], n[6] };
            for (int k = i + 8; k < g->n_nodes; ++k) {
                for (int s = 0; s < GGML_MAX_SRC; ++s) {
                    if (r5_in(g->nodes[k]->src[s], skipped, 5)) {
                        ++n_bad;
                        fprintf(stderr, "r5_fmoe: node %d %s reads a skipped node of the pattern at %d\n", k, g->nodes[k]->name, i);
                    }
                }
            }
            if (++n_checked % 40 == 0) fprintf(stderr, "r5_fmoe: consumer check %d patterns, %d external reads\n", n_checked, n_bad);
        }
    }

    const struct ggml_tensor * w0 = down->src[0], * par = down->src[1], * ids = down->src[2], * ws = sd->src[0];
    const struct ggml_tensor * W = mma->src[1], * inp = r5_fmoe_inp(a1, mma);
    const struct r5_fmoe_layout l = r5_fmoe_lay(n);
    const int n_par = (int)par->ne[1], n_ids = (int)ids->ne[0];
    char * wq = (char *)params->wdata;
    const int64_t n_embd = down->ne[0];
    const int units = (int)(n_embd/16);
    const int u0 = (int)((int64_t)ith*units/nth), u1 = (int)((int64_t)(ith + 1)*units/nth);
    const int64_t r0 = 16*(int64_t)u0, nr = 16*(int64_t)(u1 - u0);

    // P0: quantize the routed inputs, snapshot ids, weights and this thread's residual slice
    // (nodes 3 and 4 may be allocated over any of them)
    {
        const int step = nth >= n_par ? nth/n_par : 1;
        for (int r = (ith % step == 0 ? ith/step : n_par); r < n_par; r += (nth >= n_par ? n_par : nth)) {
            type_traits[l.vt_r].from_float((const float *)((const char *)par->data + r*par->nb[1]), wq + l.off_r + r*l.rs_r, par->ne[0]);
        }
    }
    int32_t sel[16];
    float wts[16];
    for (int j = 0; j < n_ids; ++j) {
        sel[j] = *(const int32_t *)((const char *)ids->data + j*ids->nb[0]);
        wts[j] = *(const float *)((const char *)W->data + j*W->nb[1]);
    }
    float res[nr > 0 ? nr : 1];
    if (nr > 0) memcpy(res, (const float *)inp->data + r0, nr*sizeof(float));
    ggml_barrier(params->shared);

    // P1, P2: shared-expert gate and up/gate (independent of the routed part)
    static int keep_b = -1;
    if (keep_b < 0) { const char * e = getenv("IK_R5_FMOE_B"); keep_b = e ? atoi(e) : 1; }
    if (ggml_compute_forward(params, gt, g, i + 3) != i + 3) GGML_ABORT("r5_fmoe: gate node was fused");
    if (keep_b) ggml_barrier(params->shared);
    if (ggml_compute_forward(params, ug, g, i + 4) != i + 4) GGML_ABORT("r5_fmoe: up/gate node was fused");
    ggml_barrier(params->shared);

    // P3: quantize the shared-expert input (a thread on the second half, so another CCX/node does it)
    if (ith == nth/2) type_traits[l.vt_s].from_float((const float *)ug->data, wq + l.off_s, ug->ne[0]);
    const float x = *(const float *)gt->data;
    ggml_barrier(params->shared);

    // P4: this thread's output rows: routed + shared down rows, then the combine
    if (nr > 0) {
        const int numa_node = ggml_numa_node_for_thread(ith, nth);
        const char * a_r = (const char *)ggml_numa_tensor_data(w0, numa_node);
        const char * a_s = (const char *)ggml_numa_tensor_data(ws, numa_node);
        float buf[(n_ids + 3)*nr];
        for (int j = 0; j < n_ids; ++j) {
            const int32_t a = sel[j];
            float * c = buf + j*nr;
            if (a < 0 || a >= w0->ne[2]) { memset(c, 0, nr*sizeof(float)); continue; }
            if (!iqk_r5_rows(nr, w0->ne[0], w0->type, a_r + a*w0->nb[2] + r0*w0->nb[1], w0->nb[1],
                             l.vt_r, wq + l.off_r + (j % n_par)*l.rs_r, l.rs_r, c)) GGML_ABORT("r5_fmoe: routed type not supported");
        }
        float * sh = buf + n_ids*nr;
        if (!iqk_r5_rows(nr, ws->ne[0], ws->type, a_s + r0*ws->nb[1], ws->nb[1], l.vt_s, wq + l.off_s, l.rs_s, sh))
            GGML_ABORT("r5_fmoe: shared type not supported");
        const float gate = gated->op_params[0] == GGML_UNARY_OP_SILU ? ggml_silu_f32(x) : 1.0f/(1.0f + expf(-x));
        const float limit = *(const float *)(gated->op_params + 1);
        iqk_r5_moe_tail((int)nr, n_ids, buf, nr, wts, res, sh, gate, limit, (float *)out->data + r0, sh + nr, sh + 2*nr);
    }
    return i + 7;
}

'''
s = s[:a] + NEW + s[b:]

# MUL_MAT_ID dispatch: drop the old debug block, keep one miss line per decode graph layer in mode 3
a = s.index('        case GGML_OP_MUL_MAT_ID:\n            {\n                if (r5_fmoe_mode() > 3')
b = s.index('            } break;', a) + len('            } break;')
s = s[:a] + '''        case GGML_OP_MUL_MAT_ID:
            {
                if (r5_fmoe_mode() > 0 && r5_fmoe_match(cgraph, i, params->wsize)) {
                    i = r5_fmoe(params, cgraph, i);
                } else {
                    if (r5_fmoe_mode() == 3 && params->ith == 0 && tensor->src[2]->ne[1] == 1) {
                        static int r5f_m = 0;
                        if (r5f_m++ < 100) fprintf(stderr, "r5_fmoe: no match at node %d %s (check line %d)\\n", i, tensor->name, r5f_line);
                    }
                    ggml_compute_forward_mul_mat_id(params, tensor);
                }
            } break;''' + s[b:]
assert 'r5_fmoe ids' not in s and 'r5_fmoe: enter' not in s
p.write_text(s)
print('patched')
