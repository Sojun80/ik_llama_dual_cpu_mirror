#!/usr/bin/env python3
"""Flint patch 2: IK_R5_FA_SHADOW=1 also covers ik's GQA decode FA loop, for layers without a sliding window.

The r5 FA kernel loses on Flint (tools/apply_flintfa.NOT-KEPT.py, runs/07-fa-ab: every thread reads all 32 query heads
and keeps partials for all of them, ~45 us per call before any work). ik's loop splits by KV head and position chunk,
which suits 4 KV heads and 32 query heads, but reads each head's 256 bytes at a 1 KB stride from memory shared by all
threads. Here each thread keeps a packed private copy of its chunks, per global attention layer and per step slot of
the thread, kept current through the same log of KV writes as patch 22 of round 5. Same chunks, same order, same
kernel: only the memory K and V are read from changes, so results should be bitwise identical to ik's FA.
- Rows past n_eff are masked, never read as data: they are not copied (a new token would otherwise refill the chunk).
- Sliding-window layers are skipped: the window moves every token.
- A chunk's thread can change when the padded cache length grows (every 256 positions): that is a refill.
Memory: 1.5 x the thread's K and V chunks, rounded to 64 KiB per chunk.
"""
import sys, pathlib
root = pathlib.Path(sys.argv[1])
p = root / 'ggml/src/iqk/iqk_flash_attn.cpp'
s = p.read_text()
assert 'shadow_chunk' not in s


def sub(old, new):
    global s
    assert s.count(old) == 1, (old[:80], s.count(old))
    s = s.replace(old, new)


sub('''static bool run(int neq2, long nbq2, int nek2, long nbk2, long nbv2, long nb1, int nek1,
''', '''// same for one chunk of ik's GQA loop: rows of one KV head, read at a stride, packed here. src_k/src_v point at row j0 of
// the head. The buffer holds `rows` rows; only rows below n_eff are valid, the masked rest is never used as data.
// One entry per (layer K/V, step slot of this thread).
static bool shadow_chunk(const char * layer_k, const char * layer_v, int slot, const char * src_k, const char * src_v,
                         long stride_k, long stride_v, long row, int j0, int rows, int n_eff,
                         const char ** kb_out, const char ** vb_out) {
    struct Shadow { const char * lk, * lv, * sk, * sv; int slot; char * buf; size_t cap; int j0, j1, n_eff; uint64_t seq; };
    constexpr int kSites = 128;
    static thread_local Shadow sh[kSites];
    static thread_local int n_sh = 0;
    Shadow * e = nullptr;
    for (int i = 0; i < n_sh; ++i) if (sh[i].lk == layer_k && sh[i].lv == layer_v && sh[i].slot == slot) { e = &sh[i]; break; }
    if (!e) {
        if (n_sh == kSites) return false;
        e = &sh[n_sh++];
        *e = { layer_k, layer_v, nullptr, nullptr, slot, nullptr, 0, 0, 0, 0, 0 };
    }
    const int j1 = std::max(j0, std::min(j0 + rows, n_eff));
    const size_t need = (size_t)rows*row;   // per K or V
    bool refill = !e->buf || e->sk != src_k || e->sv != src_v || e->j0 != j0 || j1 < e->j1 || n_eff < e->n_eff || 2*need > e->cap;
    uint64_t now = 0;
    if (!refill && e->j1 > e->j0) {
        uint64_t n2 = 0;
        refill = ggml_r5kv_written(e->seq, src_k, src_k + (long)(e->j1 - 1 - j0)*stride_k + row, &now) ||
                 ggml_r5kv_written(e->seq, src_v, src_v + (long)(e->j1 - 1 - j0)*stride_v + row, &n2);
    } else {
        now = ggml_r5kv_seq();
    }
    if (2*need > e->cap) {
        if (e->buf) free(e->buf);
        const size_t half = ((need + need/2) + (1u << 16) - 1) & ~(size_t)((1u << 16) - 1);
        e->cap = 2*half;
        e->buf = (char *)aligned_alloc(4096, e->cap);
        if (!e->buf) { e->cap = 0; return false; }
        memset(e->buf, 0, e->cap);
    }
    char * kb = e->buf, * vb = e->buf + e->cap/2;
    for (int j = refill ? j0 : e->j1; j < j1; ++j) {
        memcpy(kb + (size_t)(j - j0)*row, src_k + (long)(j - j0)*stride_k, row);
        memcpy(vb + (size_t)(j - j0)*row, src_v + (long)(j - j0)*stride_v, row);
    }
    e->sk = src_k; e->sv = src_v; e->j0 = j0; e->j1 = j1; e->n_eff = n_eff; e->seq = now;
    *kb_out = kb; *vb_out = vb;
    return true;
}

static bool run(int neq2, long nbq2, int nek2, long nbk2, long nbv2, long nb1, int nek1,
''')

sub('''        for (int istep_k = ith; istep_k < nstep_k; istep_k += nth) {
            int ik02 = istep_k/nkk;
''', '''#ifdef R5FA
        // IK_R5_FA_SHADOW: read this thread's chunks from packed private copies
        const bool shadow = n_swa == 0 && mask && int_type_k == GGML_TYPE_F16 && int_type_v == GGML_TYPE_F16 && Dk == Dv &&
                            nbk2 < stride_k && nbv2 < stride_v && r5fa::shadow_enabled();
        int n_eff = nek1;
        if (shadow) {
            auto um = (const uint16_t *)mask;
            while (n_eff > 0 && um[n_eff - 1] == 0xFC00) --n_eff;
        }
        int slot = 0;
#endif
        for (int istep_k = ith; istep_k < nstep_k; istep_k += nth) {
            int ik02 = istep_k/nkk;
''')
sub('''            auto this_m = mask ? (const char *)mask + ik01*sizeof(uint16_t) : nullptr; // we don't have ggml_half available here
            if (!iqk_flash_attn_impl(int_type_k, int_type_v,
                     Dk, Dv, rk2, this_nk, nbq2, stride_k, stride_v, 0, Dv,
                     this_q, (const void *)this_k, (const void *)this_v, (const void *)this_m, nullptr, 0,
                     scale, softcap, this_result, this_result + (Dv+0)*rk2, this_result + (Dv+1)*rk2)) return false;
        }
''', '''            auto this_m = mask ? (const char *)mask + ik01*sizeof(uint16_t) : nullptr; // we don't have ggml_half available here
            int this_stride_k = stride_k, this_stride_v = stride_v;
#ifdef R5FA
            if (shadow) {
                const long row = 2*(long)Dk;
                const char * kb, * vb;
                if (r5fa::shadow_chunk((const char *)k, (const char *)v, slot++, this_k, this_v, stride_k, stride_v, row,
                                       ik01, this_nk, n_eff, &kb, &vb)) {
                    this_k = kb; this_v = vb;
                    this_stride_k = this_stride_v = (int)row;
                }
            }
#endif
            if (!iqk_flash_attn_impl(int_type_k, int_type_v,
                     Dk, Dv, rk2, this_nk, nbq2, this_stride_k, this_stride_v, 0, Dv,
                     this_q, (const void *)this_k, (const void *)this_v, (const void *)this_m, nullptr, 0,
                     scale, softcap, this_result, this_result + (Dv+0)*rk2, this_result + (Dv+1)*rk2)) return false;
        }
''')
p.write_text(s)
print('patched')
