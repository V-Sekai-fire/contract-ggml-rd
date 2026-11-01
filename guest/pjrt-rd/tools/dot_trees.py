import sys, json
sys.path.insert(0, ".")
import jax, jax.numpy as jnp, numpy as np
from reveal_sum_order import BIG, tree_from, show
def tree(M, K, N):
    W = np.ones((K, N), np.float32)
    f = jax.jit(lambda a: a @ jnp.asarray(W))
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    S = np.zeros((K, K), np.int64)
    for st in range(0, len(pairs), M):
        chunk = pairs[st:st + M]
        x = np.ones((M, K), np.float32)
        for r, (i, j) in enumerate(chunk):
            x[r, i] = BIG; x[r, j] = -BIG
        out = np.asarray(f(jnp.asarray(x)))
        for r, (i, j) in enumerate(chunk):
            if not (out[r] == out[r, 0]).all(): return "cols-differ"
            S[i, j] = S[j, i] = K - int(round(float(out[r, 0])))
    return show(tree_from(S, K))
res = {}
for K in (2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 17, 24, 32):
    res["K%d M8 N8" % K] = tree(8, K, 8)
for (M, N) in ((1, 8), (4, 8), (8, 1), (16, 16), (3, 5)):
    res["K16 M%d N%d" % (M, N)] = tree(M, 16, N)
for k, v in res.items(): print(k, "=>", v)
