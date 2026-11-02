#!/usr/bin/env python3
"""Round-5 patch 22: IK_R5_DN_SPLIT=1 splits each delta-net head's value rows over two threads at decode.

The decode profile (runs/81) shows the fused delta-net step ~1.4 ms idle per token: 32 heads (head_dim 128) over 64
threads, so half the threads have no head (max - mean thread compute 1.34 ms). Within a head every value row r is
independent: v_prime[r] and out[r] read only entry r of each state column, and the update writes only those entries;
only the scalar q.k is shared, and each part recomputes it. With n_tokens == 1, no saved steps and nth >= 2 x heads,
the kernel is instantiated with NSPLIT = 2: unit u = head u/2, value rows [r0, r0 + PART) with PART = head_dim/2,
r0 = (u%2)*PART; thread ith takes units [ith*units/nth, (ith+1)*units/nth), so the two halves of a head land on
adjacent threads of the same NUMA node (the state binding stays valid). The state, v and out pointers are offset by
r0 and the row loops keep a compile-time trip count (PART): a first version with runtime row bounds was bit-exact but
made the kernel 2.5x slower (runs/86, runs/87; microbenchmark tools/dnbench.cpp, runs/88). Every row is computed
exactly as before, so results are bit-identical.
"""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'ggml/src/iqk/iqk_mul_mat.cpp'
s = p.read_text()
assert 'IK_R5_DN_SPLIT' not in s

def sub(old, new):
    global s
    assert s.count(old) == 1, (old[:80], s.count(old))
    s = s.replace(old, new)

sub('''const R5DynConfig & r5_dyn_config() { static R5DynConfig c; return c; }
}
''', '''const R5DynConfig & r5_dyn_config() { static R5DynConfig c; return c; }
// round-5 (IK_R5_DN_SPLIT=1): split delta-net heads by value rows at decode
bool r5_dn_split() { static const bool on = [] { auto e = getenv("IK_R5_DN_SPLIT"); return e && atoi(e) > 0; }(); return on; }
}
''')

# --- x86 kernel: everything below happens inside iqk_fused_delta_net_impl only (the neon kernel has the same loops) ---
a = s.index('template <int head_dim>\nvoid iqk_fused_delta_net_impl(')
b = s.index('bool iqk_fused_delta_net(int head_dim', a)
k = s[a:b]

def ksub(old, new, count=1):
    global k
    assert k.count(old) == count, (old[:80], k.count(old))
    k = k.replace(old, new)

ksub('template <int head_dim>\nvoid iqk_fused_delta_net_impl(', 'template <int head_dim, int NSPLIT = 1>\nvoid iqk_fused_delta_net_impl(')
ksub('''    const int h_start = (int)((int64_t)ith * total_heads / nth);
    const int h_end   = (int)((int64_t)(ith + 1) * total_heads / nth);
''', '''    // round-5 (IK_R5_DN_SPLIT=1): NSPLIT > 1 cuts each head into NSPLIT parts of PART value rows;
    // unit u = head u/NSPLIT, rows [r0, r0 + PART) with r0 = (u%NSPLIT)*PART
    constexpr int PART = head_dim / NSPLIT;
    static_assert(head_dim % NSPLIT == 0 && PART % 16 == 0);
    const int units   = total_heads * NSPLIT;
    const int u_start = (int)((int64_t)ith * units / nth);
    const int u_end   = (int)((int64_t)(ith + 1) * units / nth);
''')
ksub('''    for (int h_idx = h_start; h_idx < h_end; ++h_idx) {
        const int batch_idx = h_idx / n_heads;
''', '''    for (int u = u_start; u < u_end; ++u) {
        const int h_idx = u / NSPLIT;
        const int r0 = (u % NSPLIT) * PART;
        const int batch_idx = h_idx / n_heads;
''')
ksub('''        float * state = state_out + state_head_offset;
        for (int i = 0; i < head_dim * head_dim; ++i) {
            state[i] = state_in[state_head_offset + i];
        }
''', '''        // the part's first row; columns keep the stride head_dim
        float * state = state_out + state_head_offset + r0;
        for (int col = 0; col < head_dim; ++col) {
            for (int row = 0; row < PART; ++row) {
                state[col * head_dim + row] = state_in[state_head_offset + r0 + col * head_dim + row];
            }
        }
''')
ksub('''            const float * v_t = v_data + batch_idx * vnb3 + head_idx * vnb2 + t * vnb1;
''', '''            const float * v_t = v_data + batch_idx * vnb3 + head_idx * vnb2 + t * vnb1 + r0;
''')
ksub('''            float * out_t = out_data + out_head_offset + t * out_token_stride;
''', '''            float * out_t = out_data + out_head_offset + t * out_token_stride + r0;
''')
ksub('j < head_dim/16;', 'j < PART/16;', count=4)
ksub('row < head_dim;', 'row < PART;', count=4)
s = s[:a] + k + s[b:]

# dispatcher
sub('''    if (head_dim == 64) {
        iqk_fused_delta_net_impl<64>(n_heads, gqa_ratio, repeat_type, n_tokens, n_seqs, vnb1, vnb2, vnb3, q_data, k_data, v_data, g_data, beta_data, state_in,
                out_data, state_out, saved_steps, state_step_stride, ith, nth);
    } else {
        iqk_fused_delta_net_impl<128>(n_heads, gqa_ratio, repeat_type, n_tokens, n_seqs, vnb1, vnb2, vnb3, q_data, k_data, v_data, g_data, beta_data, state_in,
                out_data, state_out, saved_steps, state_step_stride, ith, nth);
    }
''', '''#ifndef __ARM_NEON
    // round-5 (IK_R5_DN_SPLIT=1): single-token decode with at least twice as many threads as heads splits each head
    if (n_tokens == 1 && !saved_steps && nth >= 2*n_heads*n_seqs && r5_dn_split()) {
        if (head_dim == 64) {
            iqk_fused_delta_net_impl<64, 2>(n_heads, gqa_ratio, repeat_type, n_tokens, n_seqs, vnb1, vnb2, vnb3, q_data, k_data, v_data, g_data, beta_data, state_in,
                    out_data, state_out, saved_steps, state_step_stride, ith, nth);
        } else {
            iqk_fused_delta_net_impl<128, 2>(n_heads, gqa_ratio, repeat_type, n_tokens, n_seqs, vnb1, vnb2, vnb3, q_data, k_data, v_data, g_data, beta_data, state_in,
                    out_data, state_out, saved_steps, state_step_stride, ith, nth);
        }
        return true;
    }
#endif
    if (head_dim == 64) {
        iqk_fused_delta_net_impl<64>(n_heads, gqa_ratio, repeat_type, n_tokens, n_seqs, vnb1, vnb2, vnb3, q_data, k_data, v_data, g_data, beta_data, state_in,
                out_data, state_out, saved_steps, state_step_stride, ith, nth);
    } else {
        iqk_fused_delta_net_impl<128>(n_heads, gqa_ratio, repeat_type, n_tokens, n_seqs, vnb1, vnb2, vnb3, q_data, k_data, v_data, g_data, beta_data, state_in,
                out_data, state_out, saved_steps, state_step_stride, ith, nth);
    }
''')
p.write_text(s)

# the in-place note in ggml.c: threads now own disjoint (head, row) slices
p = pathlib.Path(sys.argv[1]) / 'ggml/src/ggml.c'
t = p.read_text()
old = '''        // back, and threads own disjoint heads
'''
assert t.count(old) == 1
t = t.replace(old, '''        // back, and threads own disjoint heads (or disjoint value rows of a head, IK_R5_DN_SPLIT)
''')
p.write_text(t)
print('patched')
