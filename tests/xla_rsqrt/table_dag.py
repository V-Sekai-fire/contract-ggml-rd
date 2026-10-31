import numpy as np
from floor_of_line import tables
T = tables()

def build(y, order):
    """Hash-consed DAG over 12 index bits. order = list of bit positions, top of the DAG first.
    A node at depth d describes the sub-table over the remaining bits as (child0, child1, delta),
    where delta = value(first entry of child1) - value(first entry of child0): values are relative,
    so equal shapes at different heights share a node."""
    n = len(order)
    uniq = [dict() for _ in range(n + 1)]
    def rec(depth, fixed):           # fixed: dict bit->value already chosen
        if depth == n:
            idx = sum(v << b for b, v in fixed.items())
            return 0, int(y[idx])    # leaf id 0, absolute value
        b = order[depth]
        i0, v0 = rec(depth + 1, {**fixed, b: 0})
        i1, v1 = rec(depth + 1, {**fixed, b: 1})
        key = (i0, i1, v1 - v0)
        d = uniq[depth + 1 - 1]
        nid = d.setdefault(key, len(d))
        return nid, v0
    root, base = rec(0, {})
    return [len(u) for u in uniq[:n]], root, base

for e in (127, 128):
    y = T[e]
    msb = list(range(11, -1, -1))        # split the top bit first
    lsb = list(range(0, 12))              # split the bottom bit first
    for name, order in (("high bits first", msb), ("low bits first", lsb)):
        sizes, root, base = build(y, order)
        print("e=%d %-16s unique nodes per level (root..leaves): %s  total %d" % (e, name, sizes, sum(sizes)))
