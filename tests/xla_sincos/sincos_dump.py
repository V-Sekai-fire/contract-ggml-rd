import jax, jax.numpy as jnp, numpy as np
rng = np.random.default_rng(11)
x = np.concatenate([rng.uniform(-10, 10, 400000), rng.uniform(-1000, 1000, 200000), rng.uniform(-1, 1, 200000),
    rng.integers(0, 2**32, 200000, dtype=np.uint64).astype(np.uint32).view(np.float32)]).astype(np.float32)
x = x[np.isfinite(x)]
x.tofile("sc_in.bin")
np.asarray(jax.jit(jnp.sin)(x)).tofile("sc_sin.bin")
np.asarray(jax.jit(jnp.cos)(x)).tofile("sc_cos.bin")
print(len(x))
