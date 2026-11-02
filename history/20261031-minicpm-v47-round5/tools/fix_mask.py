#!/usr/bin/env python3
# r5bench: with R5_CASEBOOK_OFF=1 the mirror server's numactl mask must include cores 0-15, else threads pinned there
# fail to move and pile onto core 16 (runs/68-INVALID-cpumask)
import pathlib
p = pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round5/tools/r5bench.py')
s = p.read_text()
old = "cmd = ['numactl', '--physcpubind=16-95', cfg['bin']"
assert s.count(old) == 1
s = s.replace(old, "cmd = ['numactl', '--physcpubind=' + ('16-95' if FORBIDDEN else '0-95'), cfg['bin']")
p.write_text(s)
print('ok')
