import jax, jax.numpy as jnp, numpy as np
rng = np.random.default_rng(5)
n = 600000
a = np.concatenate([rng.uniform(0.01, 100, n//2), np.full(n//4, 10000.0), rng.uniform(0, 5, n//4)]).astype(np.float32)
b = np.concatenate([rng.uniform(-4, 4, n//2), rng.uniform(0, 1, n//4), rng.uniform(-3, 3, n//4)]).astype(np.float32)
a.tofile("pw_a.bin"); b.tofile("pw_b.bin")
np.asarray(jax.jit(jnp.power)(a, b)).tofile("pw_out.bin")
print(len(a))
