import ctypes, os
import numpy as np
lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "xla_rsqrt.dll"))
P = ctypes.POINTER(ctypes.c_uint32)
# est path with steps=0: rsqrt_many applies zero Newton steps, so it returns the raw estimate
def est(bits):
    out = np.empty_like(bits)
    lib.rsqrt_many(bits.ctypes.data_as(P), out.ctypes.data_as(P), ctypes.c_long(len(bits)), 0, 0, 0)
    return out

if __name__ == "__main__":
    for e in (127, 128):
        idx = np.arange(4096, dtype=np.uint32)
        x = (np.uint32(e) << np.uint32(23)) | (idx << np.uint32(11))
        y = np.round(est(x).view(np.float32).astype(np.float64) * 8192).astype(np.int64)
        d = np.diff(y)
        print("e=%d first values" % e, y[:24].tolist())
        print("  diffs", d[:40].tolist())
        print("  diff histogram", {int(k): int(v) for k, v in zip(*np.unique(d, return_counts=True))})
        d2 = np.diff(d)
        print("  second-diff histogram", {int(k): int(v) for k, v in zip(*np.unique(d2, return_counts=True))})
