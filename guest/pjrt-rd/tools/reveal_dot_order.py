"""Reveal the summation tree of XLA CPU's dot over its contracted dimension (FPRev).

    python reveal_dot_order.py M K N

Row i of the left operand holds 2^25 at one position and -2^25 at another, ones elsewhere, and the right
operand is all ones; the ones that survive say how large the smallest subtree is that holds both positions.
Run it from tools/ with reveal_sum_order.py beside it (it imports the tree builder from there).
Measured (jax 0.12.0.1): K=16 with M=4, N=8 gives the same tree as a float reduce of 16 elements,
(((5+(3+4))+(2+(0+1)))+((10+(9+(8+(6+7))))+((14+15)+(13+(11+12))))), and K=24, K=32 give others.
"""
import sys
import jax, jax.numpy as jnp, numpy as np
from reveal_sum_order import BIG, tree_from, show

M, K, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
W = np.ones((K, N), np.float32)
f = jax.jit(lambda a: a @ jnp.asarray(W))
pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
S = np.zeros((K, K), np.int64)
for st in range(0, len(pairs), M):
    chunk = pairs[st:st + M]
    x = np.ones((M, K), np.float32)
    for r, (i, j) in enumerate(chunk):
        x[r, i] = BIG
        x[r, j] = -BIG
    out = np.asarray(f(jnp.asarray(x)))
    for r, (i, j) in enumerate(chunk):
        assert (out[r] == out[r, 0]).all(), "columns differ"
        S[i, j] = S[j, i] = K - int(round(float(out[r, 0])))
print("M=%d K=%d N=%d" % (M, K, N))
print(show(tree_from(S, K)))
