#!/usr/bin/env python3
# debug: remember the failing check line of r5_fmoe_match and print it with the match result
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'ggml/src/ggml.c'
s = p.read_text()
a = s.index('#define R5F_FAIL do {')
b = s.index('\n', a)
s = s[:a] + 'static int r5f_line;\n#define R5F_FAIL do { r5f_line = __LINE__; return false; } while (0)' + s[b:]
old = 'fprintf(stderr, "r5_fmoe: node %d %s match=%d\\n", i, tensor->name, (int)r5_fmoe_match(cgraph, i, params->wsize));'
assert s.count(old) == 1
s = s.replace(old, '{ int m = r5_fmoe_match(cgraph, i, params->wsize); fprintf(stderr, "r5_fmoe: node %d %s match=%d line %d\\n", i, tensor->name, m, r5f_line); }')
p.write_text(s)
print('ok')
