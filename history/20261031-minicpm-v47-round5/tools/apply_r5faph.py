#!/usr/bin/env python3
"""Round-5 patch 9: per-phase rdtsc timers inside r5fa::partial / merge (IK_R5_FA_DEBUG=1 only)."""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'ggml/src/iqk/iqk_flash_attn.cpp'
s = p.read_text()
assert 'r5ph' not in s

def sub(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)

sub('static inline size_t slot_floats(int neq2)', '''static bool dbg_enabled() {
    static const bool e = [] { auto v = getenv("IK_R5_FA_DEBUG"); return v && atoi(v) > 0; }();
    return e;
}
static thread_local uint64_t r5ph[6];   // setup, scores, softmax, V accumulate, merge, calls
static inline uint64_t r5ph_now() { return dbg_enabled() ? __rdtsc() : 0; }

static inline size_t slot_floats(int neq2)''')

sub('''    const int nh = nkvh*rk;                 // <= kMaxHeads
    alignas(32) float Qs[kMaxHeads*D];''', '''    const int nh = nkvh*rk;                 // <= kMaxHeads
    uint64_t tp = r5ph_now();
    alignas(32) float Qs[kMaxHeads*D];''')
sub('''    const long span_v = (nkvh - 1)*nbv2 + 2*D;
    for (int jb = j0; jb < j1; jb += JB) {''', '''    const long span_v = (nkvh - 1)*nbv2 + 2*D;
    if (dbg_enabled()) { uint64_t t = __rdtsc(); r5ph[0] += t - tp; tp = t; }
    for (int jb = j0; jb < j1; jb += JB) {''')
sub('''        // ---- mask + online softmax; S becomes p
        alignas(32) float mk[JB];''', '''        if (dbg_enabled()) { uint64_t t = __rdtsc(); r5ph[1] += t - tp; tp = t; }
        // ---- mask + online softmax; S becomes p
        alignas(32) float mk[JB];''')
sub('''        // ---- O[h] += sum_j p[h][j] * v_j
        for (int kh = 0; kh < nkvh; ++kh) {''', '''        if (dbg_enabled()) { uint64_t t = __rdtsc(); r5ph[2] += t - tp; tp = t; }
        // ---- O[h] += sum_j p[h][j] * v_j
        for (int kh = 0; kh < nkvh; ++kh) {''')
sub('''    for (int h = 0; h < nh; ++h) { m_out[h] = m[h]; l_out[h] = l[h]; }
}''', '''    for (int h = 0; h < nh; ++h) { m_out[h] = m[h]; l_out[h] = l[h]; }
}''')
# V phase end: at the end of each block iteration (just before the closing of the jb loop)
sub('''                    _mm256_storeu_ps(o2, c20); _mm256_storeu_ps(o2 + 8, c21); _mm256_storeu_ps(o3, c30); _mm256_storeu_ps(o3 + 8, c31);
                }
            }
        }
    }''', '''                    _mm256_storeu_ps(o2, c20); _mm256_storeu_ps(o2 + 8, c21); _mm256_storeu_ps(o3, c30); _mm256_storeu_ps(o3 + 8, c31);
                }
            }
        }
        if (dbg_enabled()) { uint64_t t = __rdtsc(); r5ph[3] += t - tp; tp = t; }
    }''')
# merge timing + report
sub('''    // merge: tasks are (head, 64-dim chunk)
    constexpr int DC = 64;''', '''    const uint64_t tm0 = r5ph_now();
    // merge: tasks are (head, 64-dim chunk)
    constexpr int DC = 64;''')
sub('''        for (int i = 0; i < DC/8; ++i) _mm256_storeu_ps(out + 8*i, _mm256_mul_ps(vn, acc[i]));
    }
    return true;
}''', '''        for (int i = 0; i < DC/8; ++i) _mm256_storeu_ps(out + 8*i, _mm256_mul_ps(vn, acc[i]));
    }
    if (dbg_enabled()) {
        r5ph[4] += __rdtsc() - tm0;
        if (++r5ph[5] % 640 == 0) {
            const double n = 640.0;
            fprintf(stderr, "r5fa-phase ith %d (positions %d): setup %.0f  scores %.0f  softmax %.0f  V %.0f  merge %.0f ticks/call\\n",
                    ith, j1 - j0, r5ph[0]/n, r5ph[1]/n, r5ph[2]/n, r5ph[3]/n, r5ph[4]/n);
            for (int i = 0; i < 5; ++i) r5ph[i] = 0;
        }
    }
    return true;
}''')
p.write_text(s)
print('patched')
