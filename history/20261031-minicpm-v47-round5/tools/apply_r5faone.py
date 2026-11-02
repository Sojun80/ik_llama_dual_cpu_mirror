#!/usr/bin/env python3
"""Round-5 diagnostic: IK_R5_FA_ONENODE=1|2 (mirror mode) gives all FA positions to the threads of node 0 (1) or
node 1 (2, block split: second half of the threads); the other node's threads idle until the merge. Tests whether
both sockets streaming their local KV copies at the same time slow each other down. Changes the split (not bit-exact)."""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / 'ggml/src/iqk/iqk_flash_attn.cpp'
s = p.read_text()
assert 'IK_R5_FA_ONENODE' not in s
old = '''    const int per = ((n_eff + nth - 1)/nth + 7) & ~7;
    const int j0 = std::min(n_eff, ith*per), j1 = std::min(n_eff, j0 + per);
'''
assert s.count(old) == 1
s = s.replace(old, '''    static const int onenode = [] { auto e = getenv("IK_R5_FA_ONENODE"); return e ? atoi(e) : 0; }();
    const int half = nth/2;
    const bool active = onenode == 0 || (onenode == 1 ? ith < half : ith >= half);
    const int wth = onenode ? half : nth, wid = onenode == 2 ? ith - half : ith;
    const int per = ((n_eff + wth - 1)/wth + 7) & ~7;
    const int j0 = active ? std::min(n_eff, wid*per) : n_eff, j1 = active ? std::min(n_eff, j0 + per) : n_eff;
''')
p.write_text(s)
print('patched')
