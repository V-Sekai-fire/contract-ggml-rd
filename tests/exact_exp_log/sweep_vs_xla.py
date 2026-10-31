import ctypes, os, sys, time
import numpy as np
import jax, jax.numpy as jnp

here = os.path.dirname(os.path.abspath(__file__))
lib = ctypes.CDLL(os.path.join(here, "xla_expl.dll"))
P = ctypes.POINTER(ctypes.c_uint32)
mode = sys.argv[1]
variants = [int(v) for v in sys.argv[2].split(",")]
fn = lib.exp_many if mode == "exp" else lib.log_many
jf = jax.jit(jnp.exp if mode == "exp" else jnp.log)
CH = 1 << 26
total = {v: 0 for v in variants}
t0 = time.time()
for base in range(0, 1 << 32, CH):
    x = np.arange(base, base + CH, dtype=np.uint64).astype(np.uint32)
    want = np.asarray(jf(jnp.asarray(x.view(np.float32)))).view(np.uint32)
    fw = want.view(np.float32)
    for v in variants:
        out = np.empty_like(x)
        fn(x.ctypes.data_as(P), out.ctypes.data_as(P), ctypes.c_long(CH), v)
        bad = (out != want) & ~(np.isnan(out.view(np.float32)) & np.isnan(fw))
        total[v] += int(bad.sum())
        if bad.any() and total[v] == int(bad.sum()):
            i = int(np.argmax(bad))
            print("first miss v=%d x=%08x xla=%08x mine=%08x" % (v, x[i], want[i], out[i]), flush=True)
    print("chunk %d/64 done %.0fs totals %s" % (base // CH + 1, time.time() - t0, total), flush=True)
print("FINAL", mode, total, flush=True)
