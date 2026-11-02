#!/usr/bin/env python3
"""Resolve the five conflicts from 3-way applying numa-mirror onto round5."""
import re, sys, pathlib
root = pathlib.Path(sys.argv[1])

def resolve(path, fn):
    s = path.read_text()
    pat = re.compile(r'^<<<<<<< ours\n(.*?)^=======\n(.*?)^>>>>>>> theirs\n', re.S | re.M)
    hunks = pat.findall(s)
    out = pat.sub(lambda m: fn(m.group(1), m.group(2)), s)
    assert '<<<<<<<' not in out and '>>>>>>>' not in out
    path.write_text(out)
    print(path.name, len(hunks), 'resolved')

def ggml(ours, theirs):
    if 'r5d_set(node_n + 1' in ours:
        return ('                const void * src0_next_data = ggml_numa_tensor_data(src0_next, numa_node);\n'
                '                r5d_set(node_n + 1, src0_next, src1);\n'
                '                const bool done_next = iqk_mul_mat_4d(src0_next->ne[1], ne11, ne00,\n')
    if 'set_cpu_thread_affinity' in ours:
        return ('    set_numa_thread_affinity(state->ith, state->shared->n_threads);\n'
                '    set_cpu_thread_affinity(cplan, state->ith); // explicit affinity takes precedence over NUMA\n')
    raise SystemExit('unexpected ggml hunk')

def llama(ours, theirs):
    # every llama.cpp hunk is an independent addition on each side: keep ours, then theirs.
    # In the state-restore hunk the shared trailing '}' closes OUR dsv4 block, so close it
    # before their resync block (whose own '}' is the shared one).
    if 'ggml_numa_tensor_resync(t)' in theirs:
        return ours + '        }\n' + theirs
    return ours + theirs

resolve(root / 'ggml/src/ggml.c', ggml)
resolve(root / 'src/llama.cpp', llama)
