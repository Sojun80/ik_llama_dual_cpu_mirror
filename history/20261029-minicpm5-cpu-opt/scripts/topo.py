"""CPU placement helpers for gigabyte5 (2x EPYC 7K62, NPS1).
Each socket: 6 CCDs x 2 CCX x 4 cores. CCX k on node n covers cores 48n+4k .. 48n+4k+3 (SMT sibling +96).
Production owns node 0 CCX 0-3 (cores 0-15); nothing here may use them."""

PROD = set(range(0, 16))


def ccx_cores(node, ccx):
    b = 48 * node + 4 * ccx
    return list(range(b, b + 4))


def spread(node, threads, ccxs=range(12), smt=False):
    """Round-robin threads over the given CCXs, alternating CCDs first (CCX 0 of each CCD, then CCX 1),
    so every CCD's Infinity Fabric link carries an equal share of the weight stream."""
    ccxs = list(ccxs)
    order = [c for c in ccxs if c % 2 == 0] + [c for c in ccxs if c % 2 == 1]
    take = {c: 0 for c in order}
    out = []
    while len(out) < threads:
        progressed = False
        for c in order:
            if len(out) >= threads:
                break
            if take[c] < 4:
                out.append(ccx_cores(node, c)[take[c]])
                take[c] += 1
                progressed = True
        if not progressed:
            raise ValueError("not enough cores")
    if smt:
        out += [c + 96 for c in out]
    return sorted(out)


def packed(node, threads, first_ccx=0):
    b = 48 * node + 4 * first_ccx
    return list(range(b, b + threads))


def cpulist(cpus):
    cpus = sorted(cpus)
    assert not (set(cpus) & PROD) and not (set(c - 96 for c in cpus if c >= 96) & PROD), "touches production cores"
    out, start, prev = [], None, None
    for c in cpus + [None]:
        if start is None:
            start = prev = c
        elif c is not None and c == prev + 1:
            prev = c
        else:
            out.append(str(start) if start == prev else "%d-%d" % (start, prev))
            start = prev = c
    return ",".join(out)


def ccd_histogram(cpus):
    h = {}
    for c in cpus:
        p = c % 96
        key = "n%d-ccd%d" % (p // 48, (p % 48) // 8)
        h[key] = h.get(key, 0) + 1
    return h
