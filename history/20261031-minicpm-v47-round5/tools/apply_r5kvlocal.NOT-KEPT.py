#!/usr/bin/env python3
"""Round-5 experiment: IK_R5_KVLOCAL=1 makes every KV copy written only by threads of its own node (mirror mode).

ggml_numa_replicate_kv_write normally spreads the fresh rows over all threads, so each node's copy is partly written
by the other socket (and the graph CPY writes the primary copy with threads of both sockets). With KVLOCAL the threads
of node n copy the rows into node n's copy, and node 0's threads re-store one word per cache line of the primary rows
so node 0 owns them. Same bytes everywhere (bit-exact); tests whether remote-written KV lines are slower to read."""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'ggml/src/ggml.c'
s = p.read_text()
assert 'IK_R5_KVLOCAL' not in s
old = '''    const char *  base0 = (const char *) m->data[0]; // node 0 aliases the root tensor's data
'''
assert s.count(old) == 1
s = s.replace(old, old + '''    static int kvlocal = -1;
    if (kvlocal < 0) { const char * e = getenv("IK_R5_KVLOCAL"); kvlocal = e ? atoi(e) : 0; }
    if (kvlocal > 0) {
        // each node's copy is written only by that node's threads; node 0 re-stores the primary rows it owns
        const int me = ggml_numa_node_for_thread(ith, nth);
        int t0 = -1, t1 = 0;
        for (int t = 0; t < nth; ++t) if (ggml_numa_node_for_thread(t, nth) == me) { if (t0 < 0) t0 = t; t1 = t + 1; }
        const int li = ith - t0, cn = t1 - t0;
        for (int64_t r = li; r < nrows; r += cn) {
            const int64_t i1 = r % n1;
            const int64_t i2 = (r / n1) % n2;
            const int64_t i3 = r / (n1*n2);
            char * row0 = (char *) dst->data + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3];
            if (me == 0) {
                volatile uint64_t * w = (volatile uint64_t *) row0;
                for (size_t q = 0; q < run/8; q += 8) w[q] = w[q];
            } else {
                memcpy((char *) m->data[me] + (size_t) (row0 - base0), row0, run);
            }
        }
        return;
    }
''')
p.write_text(s)
print('patched')
