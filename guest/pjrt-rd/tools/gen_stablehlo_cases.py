"""Writes guest/ggml_test/stablehlo_cases.inc: real StableHLO modules from JAX, their inputs, the
output XLA's CPU backend gives for them, and a mutated module per case that must not give that output.

  pixi run --manifest-path <stablehlo-gen>/pixi.toml python gen_stablehlo_cases.py <out.inc>
"""
import sys

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

BF16 = ml_dtypes.bfloat16

# Cases whose output is not bit-equal to XLA CPU's, each with what was measured. The probe still holds
# them to the functional bar and fails if one turns bit-equal, so the list can only shrink.
XFAIL = {
    "exp_f32": "XLA CPU's exp is an approximation (9 of 128 differ from the correctly rounded value), so equality needs its exact polynomial and fused multiply-add use",
    "log_f32": "XLA CPU's log is an approximation (18 of 128 differ from the correctly rounded value), so equality needs its exact polynomial and fused multiply-add use",
    "dot_batched": "XLA CPU sums this batched dot with a separate rounding per multiply and add; its unbatched dot uses fused multiply-add and ggml-rd's mul_mat does too",
    "rsqrt_f32": "XLA CPU's rsqrt is its own approximation: it differs from one over a square root in f32 in 1244 of 4096 elements and from the correctly rounded value in 737",
    "reduce_sum_random": "XLA CPU's float summation order is an irregular tree that changes with the shape, the signature of LLVM reassociating an unrolled chain of adds: tools/reveal_sum_order.py gives (((5+(3+4))+(2+(0+1)))+((10+(9+(8+(6+7))))+((14+15)+(13+(11+12))))) for 8 rows of 16 and a different tree for 32, 48 and 64 elements, so no closed-form order reproduces it and only a per-shape table could; ggml-rd sums 256 interleaved lanes",
    "mean_last": "a float sum whose XLA CPU order is not reproduced (see reduce_sum_random), then a divide",
    "softmax_f32": "contains exp, whose XLA CPU result is not the correctly rounded value, and a float sum whose XLA CPU order is not reproduced (see reduce_sum_random)",
    "rms_norm_f32": "contains a float sum whose XLA CPU order is not reproduced (see reduce_sum_random) and rsqrt (see rsqrt_f32)",
    "sign_special": "ggml's SGN returns +0 for -0, as ggml-cpu does, and XLA returns -0 (one element of 16; every other value is exact)",
    "mlp_f32": "contains exp, whose XLA CPU result is not the correctly rounded value",
}


def ints(rng, shape, lo=-4, hi=5):
    return rng.integers(lo, hi, size=shape).astype(np.float32)


def rand(rng, shape, scale=1.0):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def positive(rng, shape):
    return (rng.random(shape) * 3.0 + 0.25).astype(np.float32)


def cases():
    rng = np.random.default_rng(20261031)
    out = []

    def add(name, fn, args, mutate, note):
        out.append((name, fn, args, mutate, note))

    x, y = rand(rng, (8, 16)), rand(rng, (8, 16))
    add("add_f32", lambda a, b: a + b, [x, y], ("stablehlo.add", "stablehlo.subtract"), "elementwise")
    add("subtract_f32", lambda a, b: a - b, [x, y], ("stablehlo.subtract", "stablehlo.add"), "elementwise")
    add("multiply_f32", lambda a, b: a * b, [x, y], ("stablehlo.multiply", "stablehlo.add"), "elementwise")
    add("divide_f32", lambda a, b: a / b, [x, positive(rng, (8, 16))], ("stablehlo.divide", "stablehlo.multiply"), "elementwise")
    add("negate_f32", lambda a: -a, [x], ("stablehlo.negate", "stablehlo.abs"), "elementwise")
    add("exp_f32", lambda a: jnp.exp(a), [x], ("stablehlo.exponential", "stablehlo.negate"), "transcendental")
    add("log_f32", lambda a: jnp.log(a), [positive(rng, (8, 16))], ("stablehlo.log", "stablehlo.negate"), "transcendental")
    add("sqrt_f32", lambda a: jnp.sqrt(a), [positive(rng, (8, 16))], ("stablehlo.sqrt", "stablehlo.negate"), "elementwise")

    add("convert_f32_bf16", lambda a: a.astype(jnp.bfloat16), [x], ("stablehlo.convert", "stablehlo.negate"), "convert")
    xb = rand(rng, (8, 16)).astype(BF16)
    add("convert_bf16_f32", lambda a: a.astype(jnp.float32), [xb], ("stablehlo.convert", "stablehlo.negate"), "convert")
    add("add_bf16", lambda a, b: a + b, [xb, rand(rng, (8, 16)).astype(BF16)], ("stablehlo.add", "stablehlo.subtract"), "elementwise")

    add("broadcast_scalar_add", lambda a: a + 1.5, [x], ("stablehlo.add", "stablehlo.subtract"), "broadcast")
    add("broadcast_row_add", lambda a, r: a + r[None, :], [x, rand(rng, (16,))], ("stablehlo.add", "stablehlo.subtract"), "broadcast")
    add("broadcast_col_mul", lambda a, c: a * c[:, None], [x, rand(rng, (8,))], ("stablehlo.multiply", "stablehlo.add"), "broadcast")

    add("transpose_2d", lambda a: a.T, [x], ("dims = [1, 0]", "dims = [0, 1]"), "layout")
    add("reshape_2d", lambda a: a.reshape(16, 8), [x], ("stablehlo.reshape", "stablehlo.negate"), "layout")
    t3 = rand(rng, (2, 3, 4))
    add("transpose_3d", lambda a: jnp.transpose(a, (2, 0, 1)), [t3], ("dims = [2, 0, 1]", "dims = [1, 2, 0]"), "layout")
    add("reshape_transpose", lambda a: a.reshape(4, 6).T + 1.0, [t3], ("stablehlo.add", "stablehlo.subtract"), "layout")

    def mutate_dot(t):
        a, b = "%arg0, %arg1", "%arg1, %arg0"
        return t.replace(a, "@@", 1).replace(b, a, 1).replace("@@", b, 1) if a in t else t

    sq = (ints(rng, (8, 8)), ints(rng, (8, 8)))
    add("dot_exact", lambda a, b: a @ b, list(sq), ("DOT", "DOT"), "dot, integer-valued: every partial sum is exact")
    a, b = rand(rng, (8, 24)), rand(rng, (24, 12))
    add("dot_random", lambda p, q: p @ q, [a, b], ("OPERANDS", ""), "dot, random: reduction order is unspecified")
    add("dot_rhs_t", lambda p, q: p @ q.T, [rand(rng, (8, 24)), rand(rng, (12, 24))], ("OPERANDS", ""), "dot, rhs contracted on its last dim")
    add("dot_batched", lambda p, q: jnp.einsum("bmk,bkn->bmn", p, q), [rand(rng, (3, 6, 10)), rand(rng, (3, 10, 5))], ("OPERANDS", ""), "dot, one batch dim")
    add("dot_bf16", lambda p, q: p @ q, [rand(rng, (8, 24)).astype(BF16), rand(rng, (24, 12)).astype(BF16)], ("OPERANDS", ""), "dot, bf16 operands")
    add("mlp_f32", lambda p, w, bb: jnp.exp(p @ w + bb[None, :]), [rand(rng, (4, 16), 0.5), rand(rng, (16, 8), 0.25), rand(rng, (8,), 0.1)],
        ("stablehlo.exponential", "stablehlo.negate"), "dot, bias broadcast, exp")

    sq8 = ints(rng, (8, 8))
    add("abs_f32", lambda a: jnp.abs(a), [x], ("stablehlo.abs", "stablehlo.negate"), "exact elementwise")
    add("floor_f32", lambda a: jnp.floor(a), [rand(rng, (8, 16), 3.0)], ("stablehlo.floor", "stablehlo.negate"), "exact elementwise")
    special = np.array([[-0.0, 0.0, -1.5, 2.5, np.inf, -np.inf, 1e-30, -1e-30], [3.0, -3.0, 0.5, -0.5, 7.25, -7.25, 100.0, -100.0]], np.float32)
    add("sign_f32", lambda a: jnp.sign(a), [rand(rng, (8, 16), 3.0)], ("stablehlo.sign", "stablehlo.negate"), "exact elementwise")
    add("sign_special", lambda a: jnp.sign(a), [special], ("stablehlo.sign", "stablehlo.negate"), "sign of zeros, infinities and small values")
    add("slice_basic", lambda a: a[1:5, 2:6], [rand(rng, (6, 8))], ("[1:5, 2:6]", "[2:6, 1:5]"), "static slice")
    add("slice_strided", lambda a: a[::2, 1::3], [rand(rng, (6, 8))], ("[0:6:2, 1:8:3]", "[1:6:2, 1:8:3]"), "strided slice")
    add("slice_3d", lambda a: a[:, 1:3, ::2], [rand(rng, (2, 4, 6))], ("[0:2, 1:3, 0:6:2]", "[0:2, 0:2, 0:6:2]"), "slice of a rank-3 array")
    add("concat_dim1", lambda a, b: jnp.concatenate([a, b], axis=1), [rand(rng, (6, 8)), rand(rng, (6, 8))], ("%arg0, %arg1, dim = 1", "%arg1, %arg0, dim = 1"), "concatenate")
    add("concat_dim0_three", lambda a, b, c: jnp.concatenate([a, b, c], axis=0), [rand(rng, (2, 5)), rand(rng, (3, 5)), rand(rng, (1, 5))], ("%arg0, %arg1, %arg2", "%arg2, %arg1, %arg0"), "concatenate three operands")
    add("pad_edges", lambda a: jax.lax.pad(a, np.float32(0), ((1, 2, 0), (3, 0, 0))), [rand(rng, (6, 8))], ("low = [1, 3], high = [2, 0]", "low = [2, 3], high = [1, 0]"), "pad with zeros")
    add("reduce_sum_exact", lambda a: a.sum(axis=-1), [sq8], ("dimensions = [1]", "dimensions = [0]"), "reduce add, integer-valued: every partial sum is exact")
    add("reduce_sum_first", lambda a: a.sum(axis=0), [ints(rng, (8, 8))], ("dimensions = [0]", "dimensions = [1]"), "reduce add over the leading dimension")
    cube = ints(rng, (3, 3, 3))
    add("reduce_sum_multi", lambda a: a.sum(axis=(0, 2)), [cube], ("dimensions = [0, 2]", "dimensions = [0, 1]"), "reduce add over two non-adjacent dimensions")
    add("reduce_sum_random", lambda a: a.sum(axis=-1), [rand(rng, (8, 64))], ("dimensions = [1]", "dimensions = [0]"), "reduce add, random: the summation order is unspecified")
    add("reduce_max_last", lambda a: a.max(axis=-1), [rand(rng, (8, 8))], ("dimensions = [1]", "dimensions = [0]"), "reduce maximum")
    add("reduce_max_multi", lambda a: a.max(axis=(0, 2)), [rand(rng, (3, 3, 3))], ("dimensions = [0, 2]", "dimensions = [0, 1]"), "reduce maximum over two dimensions")
    add("reduce_sum_bf16", lambda a: a.sum(axis=-1), [ints(rng, (8, 8)).astype(BF16)], ("dimensions = [1]", "dimensions = [0]"), "reduce add on bf16 operands")
    add("mean_last", lambda a: a.mean(axis=-1), [rand(rng, (8, 64))], ("stablehlo.divide", "stablehlo.multiply"), "reduce add then divide by the count")
    add("softmax_f32", lambda a: jax.nn.softmax(a, axis=-1), [rand(rng, (8, 16), 2.0)], ("stablehlo.subtract", "stablehlo.add"), "reduce maximum, exp, reduce add, divide")
    add("rms_norm_f32", lambda a: a * jax.lax.rsqrt((a * a).mean(axis=-1, keepdims=True) + 1e-6), [rand(rng, (8, 16))], ("stablehlo.rsqrt", "stablehlo.sqrt"), "reduce add, divide, rsqrt, broadcast, multiply")
    return out


def lower(fn, args):
    specs = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in args]
    return jax.jit(fn).lower(*specs).as_text()


def mutated(text, rule):
    old, new = rule
    if old == "OPERANDS":
        for line in text.split("\n"):
            if "stablehlo.dot_general" in line:
                head, tail = line.split("stablehlo.dot_general ", 1)
                ops, rest = tail.split(", ", 2)[0:2], tail.split(", ", 2)[2]
                return text.replace(line, head + "stablehlo.dot_general " + ops[1] + ", " + ops[0] + ", " + rest, 1)
        raise SystemExit("no dot_general to mutate")
    if old == "DOT":
        return mutated(text, ("OPERANDS", ""))
    if old not in text:
        raise SystemExit("mutation target %r not in module:\n%s" % (old, text))
    return text.replace(old, new, 1)


def dtype_tag(a):
    return {np.dtype(np.float32): "f32", np.dtype(BF16): "bf16"}[a.dtype]


def c_bytes(a):
    raw = np.ascontiguousarray(a).tobytes()
    return ",".join("0x%02x" % v for v in raw)


def main():
    path = sys.argv[1]
    lines = ["// Generated by gen_stablehlo_cases.py with jax %s. Do not edit." % jax.__version__, ""]
    names = []
    for idx, (name, fn, args, rule, note) in enumerate(cases()):
        text = lower(fn, args)
        res = np.asarray(jax.jit(fn)(*[jnp.asarray(a) for a in args]))
        bad = mutated(text, rule)
        if "stablehlo.custom_call" in text or "func.call" in text:
            raise SystemExit("%s lowered to something outside the slice:\n%s" % (name, text))
        lines.append("// %s: %s" % (name, note))
        for k, a in enumerate(args):
            lines.append("static const unsigned char %s_a%d[] = {%s};" % (name, k, c_bytes(a)))
        lines.append("static const unsigned char %s_out[] = {%s};" % (name, c_bytes(res)))
        lines.append('static const char %s_hlo[] = R"HLO(%s)HLO";' % (name, text))
        lines.append('static const char %s_bad[] = R"HLO(%s)HLO";' % (name, bad))
        argdefs = ",".join("{StablehloDtype::%s,%d,%s_a%d,sizeof(%s_a%d)}" % (dtype_tag(a).upper(), a.size, name, k, name, k) for k, a in enumerate(args))
        lines.append("static const StablehloArg %s_args[] = {%s};" % (name, argdefs))
        names.append((name, dtype_tag(res), res.size, len(args), note))
        lines.append("")
    lines.append("static const StablehloCase kStablehloCases[] = {")
    for name, tag, count, nargs, note in names:
        xf = XFAIL.get(name)
        lines.append('\t{"%s", %s_hlo, %s_bad, %s_args, %d, %s_out, sizeof(%s_out), StablehloDtype::%s, "%s", %s},' % (
            name, name, name, name, nargs, name, name, tag.upper(), note, 'nullptr' if xf is None else '"%s"' % xf))
    lines.append("};")
    open(path, "w", newline="\n").write("\n".join(lines) + "\n")
    print("wrote %s: %d cases" % (path, len(names)))


if __name__ == "__main__":
    main()
