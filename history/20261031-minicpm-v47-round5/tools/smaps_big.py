#!/usr/bin/env python3
"""List mappings >= 64 MiB of a process with their AnonHugePages (THP) coverage."""
import re, sys
pid = sys.argv[1]
cur = None
for line in open(f'/proc/{pid}/smaps'):
    if re.match(r'^[0-9a-f]+-[0-9a-f]+ ', line):
        a, b = line.split()[0].split('-')
        cur = {'hdr': line.strip(), 'size': (int(b, 16) - int(a, 16)) >> 20}
    elif cur is not None and line.startswith('AnonHugePages:'):
        ahp = int(line.split()[1]) >> 10
        if cur['size'] >= 64:
            print('%7d MiB  AnonHuge %7d MiB  %s' % (cur['size'], ahp, cur['hdr'][:70]))
