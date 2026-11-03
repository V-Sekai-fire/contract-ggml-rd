"""XLA CPU's maximum, minimum, compare and select on special values: which zero, which NaN, denormals."""
import itertools
import jax, jax.numpy as jnp, numpy as np

sp = np.array([0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan, 1e-45, -1e-45, 1e-39, 1.17549435e-38, 2.0, -2.0, 3.4028235e38], np.float32)
a, b = (np.array(x, np.float32) for x in zip(*itertools.product(sp, sp)))
neg_nan = np.array([0xFFC00000, 0x7FC00001, 0x7F800001], np.uint32).view(np.float32)
h = lambda x: np.asarray(x).view(np.uint32)

def show(name, f):
    r = h(jax.jit(f)(a, b))
    return name, r

res = dict(show(n, f) for n, f in [
    ("max", jnp.maximum), ("min", jnp.minimum),
    ("gt", lambda x, y: (x > y).astype(jnp.float32)), ("ge", lambda x, y: (x >= y).astype(jnp.float32)),
    ("lt", lambda x, y: (x < y).astype(jnp.float32)), ("le", lambda x, y: (x <= y).astype(jnp.float32)),
    ("eq", lambda x, y: (x == y).astype(jnp.float32)), ("ne", lambda x, y: (x != y).astype(jnp.float32)),
    ("where_gt", lambda x, y: jnp.where(x > y, x, y)), ("relu", lambda x, y: jax.nn.relu(x)),
    ("clip", lambda x, y: jnp.clip(x, -1.0, 1.0)),
])

def model_max(x, y, is_min=False):
    # Hypothesis: DAZ inputs, NaN if either is NaN (quiet 7fc00000), -0 < +0, FTZ output.
    def daz(v):
        u = np.float32(v).view(np.uint32)
        return np.uint32(u & 0x80000000).view(np.float32) if (u & 0x7F800000) == 0 else np.float32(v)
    x, y = daz(x), daz(y)
    if np.isnan(x) or np.isnan(y):
        return np.uint32(0x7FC00000)
    if x == y:
        sx, sy = np.signbit(x), np.signbit(y)
        if is_min:
            return (x if sx else y).view(np.uint32)
        return (y if sx else x).view(np.uint32)
    r = (x if x < y else y) if is_min else (x if x > y else y)
    return np.float32(r).view(np.uint32)

for name, is_min in [("max", False), ("min", True)]:
    want = np.array([model_max(x, y, is_min) for x, y in zip(a, b)], np.uint32)
    bad = np.nonzero(want != res[name])[0]
    print(name, "model differs at", len(bad), "of", len(a))
    for i in bad[:12]:
        print("   ", hex(h(a[i])), hex(h(b[i])), "xla", hex(res[name][i]), "model", hex(want[i]))

def daz(v):
    u = np.float32(v).view(np.uint32)
    return 0.0 if (u & 0x7F800000) == 0 else float(v)
ops = {"gt": lambda x, y: x > y, "ge": lambda x, y: x >= y, "lt": lambda x, y: x < y, "le": lambda x, y: x <= y, "eq": lambda x, y: x == y, "ne": lambda x, y: x != y}
for name, f in ops.items():
    want_daz = np.array([1.0 if f(daz(x), daz(y)) else 0.0 for x, y in zip(a, b)], np.float32).view(np.uint32)
    want_raw = np.array([1.0 if f(float(x), float(y)) else 0.0 for x, y in zip(a, b)], np.float32).view(np.uint32)
    print(name, "differs from DAZ compare at", int((want_daz != res[name]).sum()), "and from raw compare at", int((want_raw != res[name]).sum()))
print("relu", [hex(v) for v in res["relu"][:: len(sp)]])
print("clip", [hex(v) for v in res["clip"][:: len(sp)]])
print("where_gt == max?", int((res["where_gt"] != res["max"]).sum()), "differ")
print("nan payloads max:", [hex(v) for v in h(jax.jit(jnp.maximum)(neg_nan, np.ones(3, np.float32)))])
print("nan payloads where:", [hex(v) for v in h(jax.jit(lambda x, y: jnp.where(x > y, y, x))(neg_nan, np.ones(3, np.float32)))])
print(jax.jit(jnp.maximum).lower(a, b).compile().as_text().split("ENTRY")[0][-600:])
