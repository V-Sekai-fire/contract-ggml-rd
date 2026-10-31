import numpy as np
from estimate_table import est

def tables():
    out = {}
    for e in (127, 128):
        idx = np.arange(4096, dtype=np.uint32)
        x = (np.uint32(e) << np.uint32(23)) | (idx << np.uint32(11))
        out[e] = np.round(est(x).view(np.float32).astype(np.float64) * 8192).astype(np.int64)
    return out

def feasible(y):
    """exists (a,b): y_r <= a + b r < y_r + 1 for all r?  maximise margin over b by ternary search."""
    r = np.arange(len(y), dtype=np.float64)
    def margin(b):
        return np.min(y + 1 - b * r) - np.max(y - b * r)
    lo, hi = -4.0, 4.0
    for _ in range(200):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if margin(m1) < margin(m2): lo = m1
        else: hi = m2
    return margin((lo + hi) / 2), (lo + hi) / 2

if __name__ == "__main__":
    T = tables()
    for e in (127, 128):
        y = T[e]
        for G in (4, 8, 16, 32, 64, 128, 256):
            ok = 0
            n = 4096 // G
            for j in range(n):
                m, b = feasible(y[j * G:(j + 1) * G])
                ok += m > 0
            print("e=%d G=%3d: %d of %d segments are floor(a + b i)" % (e, G, ok, n))
