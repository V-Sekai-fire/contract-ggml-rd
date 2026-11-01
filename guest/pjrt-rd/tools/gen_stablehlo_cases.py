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
    "dot_batched": "XLA CPU's dot accumulates in a shape-dependent tree (tools/reveal_dot_order.py), which ggml-rd's mul_mat does not reproduce for this shape",
    "reduce_sum_random": "XLA CPU's float summation order is an irregular tree that changes with the shape, the signature of LLVM reassociating an unrolled chain of adds: tools/reveal_sum_order.py gives (((5+(3+4))+(2+(0+1)))+((10+(9+(8+(6+7))))+((14+15)+(13+(11+12))))) for 8 rows of 16 and a different tree for 32, 48 and 64 elements, so no closed-form order reproduces it and only a per-shape table could; ggml-rd sums 256 interleaved lanes",
    "mean_last": "a float sum whose XLA CPU order is not reproduced (see reduce_sum_random), then a divide",
    "softmax_f32": "contains a float sum whose XLA CPU order is not reproduced (see reduce_sum_random); softmax_pinned, the same softmax with the sum order written out, is bit-equal",
    "rms_norm_f32": "contains a float sum whose XLA CPU order is not reproduced (see reduce_sum_random); pinned_rms_scale shows the sum and sqrt part is bit-equal when the order is written out",
    "mlp_pre": "XLA CPU's dot (a YNNPACK fusion) accumulates in a shape-dependent tree: for 4x16 @ 16x8 it is the same tree as a float reduce of 16, and a sequential fused multiply-add chain matches 14 of 32 elements; the bias add after it is exact (dot then add equals XLA in 32 of 32)",
    "mlp_f32": "its dot is not bit-equal (see mlp_pre); the exp after it is exact",
}


def ints(rng, shape, lo=-4, hi=5):
    return rng.integers(lo, hi, size=shape).astype(np.float32)


def rand(rng, shape, scale=1.0):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def positive(rng, shape):
    return (rng.random(shape) * 3.0 + 0.25).astype(np.float32)


def tree_sum(a):
    """Sum the last axis in a pinned order: lane i plus lane i + n/2, halving until one is left.

    The adds are explicit slices added together. A chain of reduces would not hold: XLA's
    simplifier merges it into one reduce-window and the order changes, because StableHLO leaves
    the order of a reduce unspecified. An add of two slices it keeps as written.
    """
    while a.shape[-1] > 1:
        h = a.shape[-1] // 2
        a = a[..., :h] + a[..., h:]
    return a[..., 0]


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
    # exp and log are XLA's own polynomials (polynomial_approximations.cc); these hold every input class the
    # model was checked on over all 2^32 patterns: the whole range, both clamp edges, the subnormal
    # boundary, zeros, subnormals, infinities and a NaN.
    srng = np.random.default_rng(20261031)
    bits = lambda n: srng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32).view(np.float32)
    exp_x = np.concatenate([bits(1400), srng.uniform(-90, 90, 1400), srng.uniform(-88.0, -86.5, 600), srng.uniform(87.5, 89.5, 600),
                            np.array([0.0, -0.0, 1e-45, -1e-45, 1e-39, np.inf, -np.inf, np.nan, 88.72283, 88.722839, -87.33654, -103.97], np.float32)]).astype(np.float32)
    exp_x = np.concatenate([exp_x, np.zeros(4096 - len(exp_x), np.float32)])
    add("exp_stress", lambda a: jnp.exp(a), [exp_x], ("stablehlo.exponential", "stablehlo.negate"), "XLA's exp polynomial over every input class")
    log_x = np.concatenate([bits(1800), srng.uniform(0, 200, 1200), srng.uniform(0.5, 2.0, 800), np.float32(2.0) ** srng.integers(-126, 127, 200).astype(np.float32),
                            np.array([0.0, -0.0, 1e-45, -1e-45, 1e-39, np.inf, -np.inf, np.nan, 1.0, 0.70710677, 0.70710683, -1.0], np.float32)]).astype(np.float32)
    log_x = np.concatenate([log_x, np.ones(4096 - len(log_x), np.float32)])
    add("log_stress", lambda a: jnp.log(a), [log_x], ("stablehlo.log", "stablehlo.negate"), "XLA's log polynomial over every input class")
    rs_x = np.concatenate([bits(1500), srng.uniform(0, 100, 1200), srng.uniform(1e-3, 4.0, 900), np.float32(2.0) ** srng.integers(-126, 127, 300).astype(np.float32),
                           np.array([0.0, -0.0, 1e-45, -1e-45, 1e-39, 1.1754944e-38, np.inf, -np.inf, np.nan, 1.0, 4.0, -1.0], np.float32)]).astype(np.float32)
    rs_x = np.concatenate([rs_x, np.ones(4096 - len(rs_x), np.float32)])
    sc_x = np.concatenate([srng.uniform(-10, 10, 800), srng.uniform(-1000, 1000, 400), srng.uniform(-1, 1, 400), bits(1200),
                           srng.uniform(1.0e7, 1.0e10, 500), srng.uniform(-1.0e10, -1.0e7, 200),
                           np.array([0.0, -0.0, 1e-45, 1e-5, 3.14159274, 1.57079637, -1.57079637, 6.28318548, 0.78539819, 0.785398185,
                                     16777302.0, 16777216.0, 1e6, 3370000000.0, 3.3e9, 3.4e9, 1e30, 3.4028235e38, np.inf, -np.inf, np.nan, 1e-30], np.float32)]).astype(np.float32)
    sc_x = np.concatenate([sc_x, srng.uniform(-100, 100, 4096 - len(sc_x))]).astype(np.float32)[:4096]
    add("sin_stress", lambda a: jnp.sin(a), [sc_x], ("stablehlo.sine", "stablehlo.negate"), "XLA CPU's sin (the C runtime's sinf) over every range and edge input")
    add("cos_stress", lambda a: jnp.cos(a), [sc_x], ("stablehlo.cosine", "stablehlo.negate"), "XLA CPU's cos (the C runtime's cosf) over every range and edge input")
    add("rsqrt_stress", lambda a: jax.lax.rsqrt(a), [rs_x], ("stablehlo.rsqrt", "stablehlo.negate"), "XLA CPU's rsqrt on a Zen 2 over every input class")
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
    special = np.array([[-0.0, 0.0, -1.5, 2.5, np.inf, -np.inf, 1e-30, -1e-30], [3.0, -3.0, np.nan, -0.5, 7.25, -7.25, 1e-45, -1e-45]], np.float32)
    add("sign_f32", lambda a: jnp.sign(a), [rand(rng, (8, 16), 3.0)], ("stablehlo.sign", "stablehlo.negate"), "exact elementwise")
    add("sign_special", lambda a: jnp.sign(a), [special], ("stablehlo.sign", "stablehlo.negate"), "sign of zeros, infinities, a NaN, small values and denormals")
    add("recip_special", lambda a: 1.0 / a, [special], ("stablehlo.divide", "stablehlo.multiply"), "one over zeros, infinities and small values")
    add("slice_basic", lambda a: a[1:5, 2:6], [rand(rng, (6, 8))], ("[1:5, 2:6]", "[2:6, 1:5]"), "static slice")
    add("slice_strided", lambda a: a[::2, 1::3], [rand(rng, (6, 8))], ("[0:6:2, 1:8:3]", "[1:6:2, 1:8:3]"), "strided slice")
    add("slice_3d", lambda a: a[:, 1:3, ::2], [rand(rng, (2, 4, 6))], ("[0:2, 1:3, 0:6:2]", "[0:2, 0:2, 0:6:2]"), "slice of a rank-3 array")
    add("concat_dim1", lambda a, b: jnp.concatenate([a, b], axis=1), [rand(rng, (6, 8)), rand(rng, (6, 8))], ("%arg0, %arg1, dim = 1", "%arg1, %arg0, dim = 1"), "concatenate")
    add("concat_dim0_three", lambda a, b, c: jnp.concatenate([a, b, c], axis=0), [rand(rng, (2, 5)), rand(rng, (3, 5)), rand(rng, (1, 5))], ("%arg0, %arg1, %arg2", "%arg2, %arg1, %arg0"), "concatenate three operands")
    add("pad_edges", lambda a: jax.lax.pad(a, np.float32(0), ((1, 2, 0), (3, 0, 0))), [rand(rng, (6, 8))], ("low = [1, 3], high = [2, 0]", "low = [2, 3], high = [1, 0]"), "pad with zeros")
    add("pinned_sum_tree", lambda a: tree_sum(a) + 1.0, [rand(rng, (8, 64))], ("stablehlo.add %", "stablehlo.subtract %"), "float sum with the order pinned by explicit slice adds")
    add("pinned_mean", lambda a: tree_sum(a) / 64.0, [rand(rng, (8, 64))], ("stablehlo.divide", "stablehlo.multiply"), "mean with the order pinned")
    add("pinned_rms_scale", lambda a: jnp.sqrt(tree_sum(a * a) / 64.0 + 1e-6), [rand(rng, (8, 64))], ("stablehlo.sqrt", "stablehlo.negate"), "sqrt(mean(a*a) + eps) with the order pinned; dividing by it is rewritten to rsqrt by XLA")
    add("reduce_sum_exact", lambda a: a.sum(axis=-1), [sq8], ("dimensions = [1]", "dimensions = [0]"), "reduce add, integer-valued: every partial sum is exact")
    add("reduce_sum_first", lambda a: a.sum(axis=0), [ints(rng, (8, 8))], ("dimensions = [0]", "dimensions = [1]"), "reduce add over the leading dimension")
    cube = ints(rng, (3, 3, 3))
    add("reduce_sum_multi", lambda a: a.sum(axis=(0, 2)), [cube], ("dimensions = [0, 2]", "dimensions = [0, 1]"), "reduce add over two non-adjacent dimensions")
    add("reduce_sum_random", lambda a: a.sum(axis=-1), [rand(rng, (8, 64))], ("dimensions = [1]", "dimensions = [0]"), "reduce add, random: the summation order is unspecified")
    add("reduce_max_last", lambda a: a.max(axis=-1), [rand(rng, (8, 8))], ("dimensions = [1]", "dimensions = [0]"), "reduce maximum")
    add("reduce_max_multi", lambda a: a.max(axis=(0, 2)), [rand(rng, (3, 3, 3))], ("dimensions = [0, 2]", "dimensions = [0, 1]"), "reduce maximum over two dimensions")
    add("reduce_sum_bf16", lambda a: a.sum(axis=-1), [ints(rng, (8, 8)).astype(BF16)], ("dimensions = [1]", "dimensions = [0]"), "reduce add on bf16 operands")
    add("mean_last", lambda a: a.mean(axis=-1), [rand(rng, (8, 64))], ("stablehlo.divide", "stablehlo.multiply"), "reduce add then divide by the count")
    add("mlp_pre", lambda p, w, bb: p @ w + bb[None, :], [rand(rng, (4, 16), 0.5), rand(rng, (16, 8), 0.25), rand(rng, (8,), 0.1)], ("stablehlo.dot_general", "stablehlo.subtract"), "dot plus bias, the part of mlp_f32 before exp")
    add("softmax_pinned", lambda a: (lambda e: e / tree_sum(e)[:, None])(jnp.exp(a - jnp.max(a, axis=-1, keepdims=True))), [rand(rng, (8, 16), 2.0)], ("stablehlo.subtract", "stablehlo.add"), "softmax with the sum order pinned")
    add("divide_by_const3", lambda a: a / 3.0, [rand(rng, (8, 16), 2.0)], ("stablehlo.divide", "stablehlo.multiply"), "XLA rewrites a / broadcast(3) as a * broadcast(1/3)")
    add("divide_by_row", lambda a, b: a / b[:, None], [rand(rng, (8, 16), 2.0), positive(rng, (8,))], ("stablehlo.divide", "stablehlo.multiply"), "a / broadcast(b), b a vector")
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
