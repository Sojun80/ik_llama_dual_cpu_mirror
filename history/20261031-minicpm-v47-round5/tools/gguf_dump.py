#!/usr/bin/env python3
"""Dependency-free GGUF header dump: metadata + tensor table + per-token byte model."""
import struct, sys, re, json, collections

# ggml type -> (block elems, block bytes)
TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
         6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
         10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
         13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
         30: ("BF16", 1, 2)}


def rd(f, fmt):
    s = struct.calcsize("<" + fmt)
    return struct.unpack("<" + fmt, f.read(s))


def rstr(f):
    (n,) = rd(f, "Q")
    return f.read(n).decode("utf-8", "replace")


def rval(f, t):
    if t == 0: return rd(f, "B")[0]
    if t == 1: return rd(f, "b")[0]
    if t == 2: return rd(f, "H")[0]
    if t == 3: return rd(f, "h")[0]
    if t == 4: return rd(f, "I")[0]
    if t == 5: return rd(f, "i")[0]
    if t == 6: return rd(f, "f")[0]
    if t == 7: return bool(rd(f, "B")[0])
    if t == 8: return rstr(f)
    if t == 9:
        (et, n) = rd(f, "IQ")
        if et == 8 and n > 64:
            for _ in range(n): rstr(f)
            return f"<{n} strings>"
        if n > 4096:
            sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[et]
            f.seek(n * sz, 1)
            return f"<{n} values>"
        return [rval(f, et) for _ in range(n)]
    if t == 10: return rd(f, "Q")[0]
    if t == 11: return rd(f, "q")[0]
    if t == 12: return rd(f, "d")[0]
    raise ValueError(t)


def main(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    (ver, nt, nkv) = rd(f, "IQQ")
    meta = {}
    for _ in range(nkv):
        k = rstr(f)
        (t,) = rd(f, "I")
        meta[k] = rval(f, t)
    tensors = []
    for _ in range(nt):
        name = rstr(f)
        (nd,) = rd(f, "I")
        dims = list(rd(f, "Q" * nd))
        (tt, off) = rd(f, "IQ")
        tn, be, bb = TYPES.get(tt, (f"T{tt}", 1, 0))
        ne = 1
        for d in dims: ne *= d
        tensors.append(dict(name=name, type=tn, dims=dims, bytes=ne // be * bb, offset=off))
    return meta, tensors


if __name__ == "__main__":
    meta, tensors = main(sys.argv[1])
    for k, v in meta.items():
        if k.startswith("tokenizer.") and not isinstance(v, (int, float, str)): continue
        print(f"META {k} = {v}")
    seen = collections.OrderedDict()
    for t in tensors:
        m = re.match(r"blk\.(\d+)\.(.*)", t["name"])
        if m:
            key = (m.group(2), t["type"], tuple(t["dims"]))
            seen.setdefault(key, []).append(int(m.group(1)))
        else:
            print(f"TENSOR {t['name']:40s} {t['type']:5s} {t['dims']} {t['bytes']/2**20:9.2f} MiB")
    for (n, ty, d), layers in seen.items():
        ls = layers if len(layers) < 12 else f"{len(layers)} layers [{layers[0]}..{layers[-1]}]"
        print(f"BLK {n:32s} {ty:5s} {list(d)} {ls}")
    if len(sys.argv) > 2:
        json.dump(dict(meta={k: v for k, v in meta.items() if not k.startswith("tokenizer.")},
                       tensors=tensors), open(sys.argv[2], "w"), indent=1)
