#!/usr/bin/env python3
"""Round-5 patch 13: column-split MUL_MULTI_ADD when rows < threads (decode), env IK_R5_MMA_COLS (default 1).

Single-token decode has one row, so the row split ran the weighted sum of the 8 routed-expert
outputs on thread 0 alone, which had to pull 64 KB written by every other core. Now each thread takes
a 64-byte aligned column slice over all experts. Per element the operations and their order are
unchanged, so results are bitwise identical.
"""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'ggml/src/iqk/iqk_cpu_ops.cpp'
s = p.read_text()
assert 'IK_R5_MMA_COLS' not in s

def sub(old, new, n=1):
    global s
    assert s.count(old) == n, (old[:70], s.count(old))
    s = s.replace(old, new)

sub('''    int nrows = dst->ne[1];
    int npt   = (nrows + nth - 1)/nth;
    int first = ith*npt;
    int last  = std::min(nrows, first + npt);

    int ne01 = src0->ne[1];
    int ne00 = src0->ne[0];
''', '''    int nrows = dst->ne[1];
    int npt   = (nrows + nth - 1)/nth;
    int first = ith*npt;
    int last  = std::min(nrows, first + npt);

    int ne01 = src0->ne[1];
    int ne00 = src0->ne[0];

    // round-5: with fewer rows than threads (single-token decode) split the columns instead
    int k0 = 0, k1 = ne00;
    static const bool r5_cols = [] { auto e = getenv("IK_R5_MMA_COLS"); return e ? atoi(e) > 0 : true; }();
    if (r5_cols && nrows < nth) {
        const int kpt = ((ne00 + nth - 1)/nth + 15) & ~15;   // multiple of 16 floats = one cache line
        k0 = std::min(ne00, ith*kpt);
        k1 = std::min(ne00, k0 + kpt);
        first = 0; last = k1 > k0 ? nrows : 0;
    }
''')
sub('''            float s = scales[ids[0]] * x1[0];
            for (int k = 0; k < ne00; ++k) y[k] = x0[k] * s;''', '''            float s = scales[ids[0]] * x1[0];
            for (int k = k0; k < k1; ++k) y[k] = x0[k] * s;''')
sub('''                s  = x1[0] * scales[ids[j]];
                for (int k = 0; k < ne00; ++k) y[k] += x0[k] * s;''', '''                s  = x1[0] * scales[ids[j]];
                for (int k = k0; k < k1; ++k) y[k] += x0[k] * s;''')
sub('''        for (int k = 0; k < ne00; ++k) y[k] = x0[k] * x1[0];
        for (int j = 1; j < ne01; ++j) {
            c0 += src0->nb[1];
            c1 += src1->nb[1];
            x0 = (const float *)c0;
            x1 = (const float *)c1;
            for (int k = 0; k < ne00; ++k) y[k] += x0[k] * x1[0];''', '''        for (int k = k0; k < k1; ++k) y[k] = x0[k] * x1[0];
        for (int j = 1; j < ne01; ++j) {
            c0 += src0->nb[1];
            c1 += src1->nb[1];
            x0 = (const float *)c0;
            x1 = (const float *)c1;
            for (int k = k0; k < k1; ++k) y[k] += x0[k] * x1[0];''')
p.write_text(s)
print('patched')
