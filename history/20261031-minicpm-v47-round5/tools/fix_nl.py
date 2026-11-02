#!/usr/bin/env python3
# repair string literals in ggml.c that got a raw newline instead of \n (r5_fmoe debug prints)
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'ggml/src/ggml.c'
s = p.read_text()
n = 0
for key in ('check line %d', 'active (first pattern at node %d %s)'):
    bad = key + '\n'
    while bad + '"' in s or bad + '", ' in s:
        s = s.replace(bad, key + '\\n', 1); n += 1
p.write_text(s)
print('fixed', n)
