import numpy as np
from floor_of_line import tables
T = tables()                                  # y in units 2^-13, index 12 bits, binades 127 (odd) and 128 (even)
f32 = np.float32
def base(parity_scale, idx):
    # exact f32 arithmetic, identical on host and GPU (sqrt and divide are correctly rounded)
    x = (f32(1.0) + (idx.astype(f32) * f32(2.0) + f32(1.0)) / f32(8192.0)) * f32(parity_scale)
    v = f32(8192.0) / np.sqrt(x).astype(f32)
    return np.floor(v + f32(0.5)).astype(np.int64)
idx = np.arange(4096)
res = []
for scale, e in ((1.0, 127), (2.0, 128)):
    r = T[e] - base(scale, idx)
    print("scale", scale, "residual histogram", {int(k): int(v) for k, v in zip(*np.unique(r, return_counts=True))})
    res.append(r)
res = np.concatenate(res)                     # n = parity_index*4096 + idx; parity 0 = odd exponent (scale 1), 1 = even (scale 2)
assert res.min() >= -1 and res.max() <= 1
code = (res + 1).astype(np.uint32)            # 0,1,2
words = np.zeros(512, np.uint32)
for n, c in enumerate(code):
    words[n >> 4] |= c << np.uint32(2 * (n & 15))
np.save("zen2_words.npy", words)
print("words", len(words), "first", [hex(int(w)) for w in words[:4]])
