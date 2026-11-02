#!/usr/bin/env python3
# debug: trace r5_fmoe entry and match results (mode 4)
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'ggml/src/ggml.c'
s = p.read_text()
old = '''    const int ith = params->ith, nth = params->nth;

    if (r5_fmoe_mode() > 1 && ith == 0) {'''
new = '''    const int ith = params->ith, nth = params->nth;
    if (r5_fmoe_mode() > 3 && ith == 0) { static int r5f_e = 0; if (r5f_e++ < 5) fprintf(stderr, "r5_fmoe: enter node %d %s\\n", i, down->name); }

    if (r5_fmoe_mode() > 1 && ith == 0) {'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''                if (r5_fmoe_mode() > 0 && r5_fmoe_match(cgraph, i, params->wsize)) {'''
new = '''                if (r5_fmoe_mode() > 3 && params->ith == 0 && tensor->src[2]->ne[1] == 1) {
                    static int r5f_m = 0;
                    if (r5f_m++ < 45) fprintf(stderr, "r5_fmoe: node %d %s match=%d\\n", i, tensor->name, (int)r5_fmoe_match(cgraph, i, params->wsize));
                }
                if (r5_fmoe_mode() > 0 && r5_fmoe_match(cgraph, i, params->wsize)) {'''
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s)
print('ok')
