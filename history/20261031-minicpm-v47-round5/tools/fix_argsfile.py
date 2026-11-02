#!/usr/bin/env python3
# r5bench: R5_ARGSFILE=<file> in a config's env appends extra server arguments, one per line (e.g. a draft model and
# --spec-type for speculative decoding), since ':' and ',' in those values clash with the --config syntax
import pathlib
p = pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round5/tools/r5bench.py')
s = p.read_text()
assert 'R5_ARGSFILE' not in s
old = "    mirror = env.pop('R5_MIRROR', 'weights')\n"
assert s.count(old) == 1
s = s.replace(old, old + "    argsfile = env.pop('R5_ARGSFILE', '')\n"
                         "    extra = [l.strip() for l in pathlib.Path(argsfile).read_text().splitlines() if l.strip()] if argsfile else []\n")
old = "    return dict(name=name, bin=binary, threads=threads, cpus=cpus, env=env, mode=mode, mirror=mirror)\n"
assert s.count(old) == 1
s = s.replace(old, "    return dict(name=name, bin=binary, threads=threads, cpus=cpus, env=env, mode=mode, mirror=mirror, extra=extra)\n")
old = "    base = f'http://127.0.0.1:{PORT}'\n"
assert s.count(old) == 1
s = s.replace(old, "    cmd += cfg.get('extra', [])\n" + old)
p.write_text(s)
print('ok')
