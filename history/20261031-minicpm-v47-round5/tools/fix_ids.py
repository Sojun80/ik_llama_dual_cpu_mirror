#!/usr/bin/env python3
# replace the broken ids debug print in r5_fmoe_match with a well-formed one
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'ggml/src/ggml.c'
s = p.read_text()
a = s.index('    if (r5_fmoe_mode() > 3) fprintf(stderr, "r5_fmoe ids:')
b = s.index('\n', s.index('(long)down->ne[1]);', a)) + 1
s = s[:a] + ('    if (r5_fmoe_mode() > 3 && ids->ne[1] == 1) fprintf(stderr, "r5_fmoe ids: type %d ne %ld %ld down ne1 %ld par %ld %ld %ld\\n",'
             ' ids->type, (long)ids->ne[0], (long)ids->ne[1], (long)down->ne[1], (long)par->ne[0], (long)par->ne[1], (long)par->ne[2]);\n') + s[b:]
p.write_text(s)
print('ok')
