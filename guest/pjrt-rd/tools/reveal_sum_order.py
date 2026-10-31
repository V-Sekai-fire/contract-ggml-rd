"""Reveal the order in which XLA CPU sums the last axis of an (R, n) f32 array.

Run: python reveal_sum_order.py R n

Put +2^25 at leaf i, -2^25 at leaf j and 1 everywhere else. The two big values cancel exactly,
and every 1 added into the partial sum that holds one of them before the two meet is absorbed,
so the result counts the ones outside the smallest subtree joining i and j. All pairs give the
whole summation tree (the method of FPRev). Each of the R rows probes one pair per call.
"""
import itertools
import sys

import jax
import jax.numpy as jnp
import numpy as np

BIG = np.float32(2.0 ** 25)


def reveal(R, n, axis_last=True):
    """Subtree size S[i][j] of the smallest subtree holding leaves i and j, for sum over the last axis of (R, n)."""
    f = jax.jit(lambda a: a.sum(axis=-1))
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    S = np.zeros((n, n), np.int64)
    for start in range(0, len(pairs), R):
        chunk = pairs[start:start + R]
        x = np.ones((R, n), np.float32)
        for r, (i, j) in enumerate(chunk):
            x[r, i] = BIG
            x[r, j] = -BIG
        out = np.asarray(f(jnp.asarray(x)))
        for r, (i, j) in enumerate(chunk):
            # ones outside the LCA subtree survive; the two big values cancel exactly
            S[i, j] = S[j, i] = n - int(round(float(out[r])))
    return S


def tree_from(S, n):
    """Build the binary-ish tree: repeatedly join the two clusters with the smallest S."""
    clusters = {i: ("L", i) for i in range(n)}
    members = {i: [i] for i in range(n)}
    nxt = n
    active = set(range(n))
    while len(active) > 1:
        best = None
        for a, b in itertools.combinations(sorted(active), 2):
            s = S[members[a][0], members[b][0]]
            if best is None or s < best[0]:
                best = (s, a, b)
        s, a, b = best
        clusters[nxt] = ("N", clusters[a], clusters[b])
        members[nxt] = members[a] + members[b]
        active -= {a, b}
        active.add(nxt)
        nxt += 1
    return clusters[next(iter(active))]


def show(t):
    if t[0] == "L":
        return str(t[1])
    return "(" + show(t[1]) + "+" + show(t[2]) + ")"


if __name__ == "__main__":
    R, n = int(sys.argv[1]), int(sys.argv[2])
    S = reveal(R, n)
    print("R=%d n=%d" % (R, n))
    print(show(tree_from(S, n)))
