import numpy as np
from floor_of_line import tables
T = tables()
best = {}
for e in (127, 128):
    y = T[e].astype(np.float64)
    scale = 2.0 ** (e - 127)
    G = 4
    n = 1024
    xmid = (1.0 + (np.arange(n + 2) - 0.0 + 0.5) / 1024.0)          # node k at the midpoint of 10-bit group k, k = 0..n+1
    xmid = (1.0 + (np.arange(-1, n + 1) + 0.5) / 1024.0) * scale     # index shifted by one: node[k+1] is group k
    exact = 8192.0 / np.sqrt(xmid)
    rounded = np.round(exact)                                         # Intel-style 13-bit rounded node
    for nodename, N in (("exact", exact), ("intel-rounded", rounded)):
        for how in ("floor", "round", "trunc+half"):
            for shift in (0.0, 0.25, 0.5, -0.25, -0.5):
                bad = 0
                for idx in range(4096):
                    k, i = divmod(idx, G)
                    pos = (i + 0.5) / G - 0.5 + shift                 # offset from the group midpoint in group units
                    nb = k + 1 + (1 if pos >= 0 else -1)
                    w = abs(pos)
                    val = N[k + 1] * (1 - w) + N[nb] * w
                    v = np.floor(val) if how == "floor" else (np.floor(val + 0.5) if how == "round" else np.floor(val + 0.5))
                    bad += int(v != y[idx])
                key = (nodename, how, shift)
                best.setdefault(key, 0)
                best[key] += bad
top = sorted(best.items(), key=lambda kv: kv[1])[:5]
for k, v in top: print("mismatches %5d of 8192 : nodes=%s %s shift=%.2f" % (v, *k))
