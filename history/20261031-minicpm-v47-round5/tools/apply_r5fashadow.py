#!/usr/bin/env python3
"""Round-5 patch 22: IK_R5_FA_SHADOW=1 - the r5 decode FA reads each thread's K/V rows from a private copy (mirror mode).

The decode FA reads KV at a fixed ~20-25 GB/s per node whatever the number of readers (8..48 per node, runs/97,
runs/98), well below the ~77 GB/s per node the same threads get from the weights. Reading the same rows from a copy in
memory the reading thread allocated and wrote itself is 2.0-2.4x faster in the FA (runs/100-102), bit-identical.
Not the cause (each tested): reader count, remote-socket writers (IK_R5_KVLOCAL), line ownership (re-storing the rows
in place, IK_R5_FA_REOWN), page size or placement (the node-1 KV copy is 100% THP, node-local, 2 MiB aligned:
runs/103-105). Most likely the DRAM bank/row pattern of 64 streams inside the same few huge pages; not proven.

- ggml.c keeps a log of KV writes: ggml_numa_replicate_kv_write (every KV cpy/dup in mirror mode) records the written
  byte range of every node copy, ggml_numa_tensor_resync (state restore, K-shift, cache clear) the whole tensor.
  ggml_r5kv_written(since, lo, hi, &now) tells whether an entry since `since` touched [lo, hi); a reader more than
  R5KV_LOG entries behind counts as touched.
- iqk r5fa: each thread keeps, per FA call site (K/V pointers: layer and node copy), a private copy of its rows
  [j0, j1) (2 MiB aligned, MADV_HUGEPAGE, touched by the thread). Rows appended since the last call are copied; the
  whole range is re-copied when its start moves, it shrinks, n_eff decreases, it outgrows the buffer, or a logged
  write touched rows already copied. partial() then reads the copy.
Only active when the NUMA mirror is (the write hooks live in the mirror code). Bit-identical by construction. Memory:
up to 2 x 2 MiB-rounded buffers per thread and attention layer (~1.3 GB at 64 threads, 10 layers, 9.4K).
"""
import sys, pathlib
root = pathlib.Path(sys.argv[1])

# ---------------- ggml.c: KV write log ----------------
p = root / 'ggml/src/ggml.c'
s = p.read_text()
assert 'r5kv_log' not in s

def sub(old, new):
    global s
    assert s.count(old) == 1, (old[:80], s.count(old))
    s = s.replace(old, new)

sub('''// Cold path only (state restore, K-shift, cache clear): re-copy node 0's bytes into every
''', '''// ---- round-5: log of KV writes for the decode-FA private K/V copies (IK_R5_FA_SHADOW, see apply_r5fashadow.py) ----
#define R5KV_LOG 4096
struct r5kv_entry { const char * a[GGML_NUMA_MAX_NODES], * b[GGML_NUMA_MAX_NODES]; };   // [a, b) per node copy
static struct r5kv_entry r5kv_log[R5KV_LOG];
static atomic_ullong r5kv_seq;

// one writer at a time: thread 0 inside a graph node, or the main thread between graphs
static void r5kv_log_push(const char * const * a, const char * const * b) {
    const unsigned long long n = atomic_load_explicit(&r5kv_seq, memory_order_relaxed);
    struct r5kv_entry * e = &r5kv_log[n % R5KV_LOG];
    for (int i = 0; i < GGML_NUMA_MAX_NODES; ++i) { e->a[i] = a[i]; e->b[i] = b[i]; }
    atomic_store_explicit(&r5kv_seq, n + 1, memory_order_release);
}

// read by iqk r5fa (iqk_flash_attn.cpp)
uint64_t ggml_r5kv_seq(void);
bool ggml_r5kv_written(uint64_t since, const void * lo, const void * hi, uint64_t * now);

uint64_t ggml_r5kv_seq(void) {
    return atomic_load_explicit(&r5kv_seq, memory_order_acquire);
}

// did a KV write logged in [since, now) touch [lo, hi) of any node copy? entries lost to the ring count as touching
bool ggml_r5kv_written(uint64_t since, const void * lo, const void * hi, uint64_t * now) {
    const uint64_t n = atomic_load_explicit(&r5kv_seq, memory_order_acquire);
    *now = n;
    if (n - since > R5KV_LOG) {
        return true;
    }
    for (uint64_t i = since; i < n; ++i) {
        const struct r5kv_entry * e = &r5kv_log[i % R5KV_LOG];
        for (int j = 0; j < GGML_NUMA_MAX_NODES; ++j) {
            if (e->a[j] && e->a[j] < (const char *) hi && (const char *) lo < e->b[j]) {
                return true;
            }
        }
    }
    return false;
}

// Cold path only (state restore, K-shift, cache clear): re-copy node 0's bytes into every
''')

sub('''    for (int n = 1; n < nodes; ++n) { // node 0 aliases tensor->data
        if (m->data[n] && m->data[n] != tensor->data) {
            memcpy(m->data[n], tensor->data, nbytes);
        }
    }
}
''', '''    for (int n = 1; n < nodes; ++n) { // node 0 aliases tensor->data
        if (m->data[n] && m->data[n] != tensor->data) {
            memcpy(m->data[n], tensor->data, nbytes);
        }
    }
    // IK_R5_FA_SHADOW: the whole tensor may have changed
    const char * a[GGML_NUMA_MAX_NODES] = { NULL }, * b[GGML_NUMA_MAX_NODES] = { NULL };
    for (int n = 0; n < nodes && n < GGML_NUMA_MAX_NODES; ++n) {
        const char * d = n == 0 ? (const char *) tensor->data : (const char *) m->data[n];
        if (d) { a[n] = d; b[n] = d + nbytes; }
    }
    r5kv_log_push(a, b);
}
''')

sub('''    const char *  base0 = (const char *) m->data[0]; // node 0 aliases the root tensor's data
''', '''    const char *  base0 = (const char *) m->data[0]; // node 0 aliases the root tensor's data
    if (ith == 0) {
        // IK_R5_FA_SHADOW: log the written byte range of every node copy
        const char * lo0 = (const char *) dst->data;
        const size_t len = (size_t) ((n1 - 1)*dst->nb[1] + (n2 - 1)*dst->nb[2] + (n3 - 1)*dst->nb[3]) + run;
        const char * a[GGML_NUMA_MAX_NODES] = { NULL }, * b[GGML_NUMA_MAX_NODES] = { NULL };
        for (int n = 0; n < nodes && n < GGML_NUMA_MAX_NODES; ++n) {
            if (m->data[n]) { a[n] = (const char *) m->data[n] + (lo0 - base0); b[n] = a[n] + len; }
        }
        r5kv_log_push(a, b);
    }
''')
p.write_text(s)

# ---------------- iqk_flash_attn.cpp: private copies ----------------
p = root / 'ggml/src/iqk/iqk_flash_attn.cpp'
s = p.read_text()
assert 'IK_R5_FA_SHADOW' not in s

sub('''static bool run(int neq2, long nbq2, int nek2, long nbk2, long nbv2, long nb1, int nek1,
''', '''// ---- IK_R5_FA_SHADOW: per-thread private copy of the thread's K/V rows of one FA call site (see apply_r5fashadow.py)
extern "C" uint64_t ggml_r5kv_seq(void);
extern "C" bool ggml_r5kv_written(uint64_t since, const void * lo, const void * hi, uint64_t * now);

static bool shadow_enabled() {
    static const bool e = [] { auto s = getenv("IK_R5_FA_SHADOW"); return s && atoi(s) > 0; }();
    return e && ggml_numa_mirror_active();
}

// private copies of K and V rows [j0, j1) (`row` bytes each, packed); false if no slot is left
static bool shadow_rows(const char * k, const char * v, long stride_k, long stride_v, long row, int j0, int j1,
                        int n_eff, const char ** kb_out, const char ** vb_out) {
    struct Shadow { const char * k, * v; char * buf; size_t cap; int j0, j1, n_eff; uint64_t seq; };
    constexpr int kSites = 64;
    static thread_local Shadow sh[kSites];
    static thread_local int n_sh = 0;
    Shadow * e = nullptr;
    for (int i = 0; i < n_sh; ++i) if (sh[i].k == k && sh[i].v == v) { e = &sh[i]; break; }
    if (!e) {
        if (n_sh == kSites) return false;
        e = &sh[n_sh++];
        *e = { k, v, nullptr, 0, 0, 0, 0, 0 };
    }
    const size_t need = (size_t)(j1 - j0)*row;   // per K or V
    bool refill = !e->buf || e->j0 != j0 || j1 < e->j1 || n_eff < e->n_eff || 2*need > e->cap;
    uint64_t now = 0;
    if (!refill && e->j1 > e->j0) {
        // rows already copied, changed by a logged KV write since?
        uint64_t n2 = 0;
        refill = ggml_r5kv_written(e->seq, k + (long)e->j0*stride_k, k + (long)(e->j1 - 1)*stride_k + row, &now) ||
                 ggml_r5kv_written(e->seq, v + (long)e->j0*stride_v, v + (long)(e->j1 - 1)*stride_v + row, &n2);
    } else {
        now = ggml_r5kv_seq();
    }
    if (2*need > e->cap) {
        if (e->buf) free(e->buf);
        const size_t half = ((need + need/2) + (1u << 20) - 1) & ~(size_t)((1u << 20) - 1);
        e->cap = 2*half;
        e->buf = (char *)aligned_alloc(2u << 20, e->cap);
        if (!e->buf) { e->cap = 0; return false; }
        madvise(e->buf, e->cap, MADV_HUGEPAGE);
        memset(e->buf, 0, e->cap);
    }
    char * kb = e->buf, * vb = e->buf + e->cap/2;
    for (int j = refill ? j0 : e->j1; j < j1; ++j) {
        memcpy(kb + (size_t)(j - j0)*row, k + (long)j*stride_k, row);
        memcpy(vb + (size_t)(j - j0)*row, v + (long)j*stride_v, row);
    }
    e->j0 = j0; e->j1 = j1; e->n_eff = n_eff; e->seq = now;
    *kb_out = kb; *vb_out = vb;
    return true;
}

''' + '''static bool run(int neq2, long nbq2, int nek2, long nbk2, long nbv2, long nb1, int nek1,
''')

sub('''    static const bool two_pass = [] { auto e = getenv("IK_R5_FA_2PASS"); return e && atoi(e) > 0; }();
    const uint64_t t0 = dbg ? __rdtsc() : 0;
    (two_pass ? partial2pass : partial)(j0, j1, nek2, rk, (const char *)q, nbq2, (const char *)k, stride_k, nbk2,
            (const char *)v, stride_v, nbv2, mk, scale, mine, mine + neq2, mine + 2*neq2);
''', '''    static const bool two_pass = [] { auto e = getenv("IK_R5_FA_2PASS"); return e && atoi(e) > 0; }();
    // IK_R5_FA_SHADOW: read this thread's rows from its private copy
    const char * kp = (const char *)k, * vp = (const char *)v;
    long sk = stride_k, sv = stride_v;
    const long row = (long)(nek2 - 1)*nbk2 + 2*D;   // bytes of one position over all KV heads
    if (j1 > j0 && row == (long)(nek2 - 1)*nbv2 + 2*D && shadow_enabled()) {
        const char * kb, * vb;
        if (shadow_rows((const char *)k, (const char *)v, stride_k, stride_v, row, j0, j1, n_eff, &kb, &vb)) {
            // partial() indexes rows from j0: rebase so that row j0 is the first private row
            kp = kb - (long)j0*row; vp = vb - (long)j0*row;
            sk = row; sv = row;
        }
    }
    const uint64_t t0 = dbg ? __rdtsc() : 0;
    (two_pass ? partial2pass : partial)(j0, j1, nek2, rk, (const char *)q, nbq2, kp, sk, nbk2,
            vp, sv, nbv2, mk, scale, mine, mine + neq2, mine + 2*neq2);
''')
p.write_text(s)
print('patched')
