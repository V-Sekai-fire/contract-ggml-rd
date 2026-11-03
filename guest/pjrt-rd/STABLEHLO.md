# StableHLO in pjrt-rd

`stablehlo.cpp` lowers one `func.func @main` of a StableHLO module to a ggml graph. Row-major shapes become ggml `ne` by reversing the dimensions. Constants carry their bytes in the `Program`, and arguments keep their declared order.

## Supported (33 ops)

`constant`, `iota`, `convert`, `reshape`, `transpose`, `broadcast_in_dim`, `add`, `subtract`, `multiply`, `divide`, `negate`, `exponential`, `log`, `sqrt`, `rsqrt`, `sine`, `cosine`, `dot_general`, `reduce`, `maximum`, `minimum`, `abs`, `floor`, `sign`, `slice`, `concatenate`, `pad`, `compare`, `select`, `and`, `or`, `xor` and `not`, and `call` (or `func.call`) of the module's private functions.

Element types are f32 and bf16 for values computed at run time. i32 and i1 values are accepted when they are known at compile time (an `iota`, a constant, and anything computed from those): they are folded on the host and never become a ggml tensor. That covers the masks JAX writes for causal attention and `jnp.where`, and an index table converted to f32. Float arithmetic is never folded, so its rounding is always the device's. Shapes are static. A rank above 4 is accepted when at most four dimensions are larger than one: the tensor keeps only those (attention's intermediates are rank 5). bf16 element-wise ops run in f32 and round back, as XLA does.

A call is inlined: the callee's statements are renamed so they cannot collide, its parameters are bound to the caller's operands, and its returned value takes the caller's name. Calls inside callees are inlined the same way; a chain deeper than 32 (a recursive function) is refused.

`dot_general` takes any number of batch, free and contracting dimensions: each group is transposed together and flattened, so the product is one `ggml_mul_mat` of `[B, M, K]` by `[B, N, K]`, and the result `[B, M, N]` is reshaped to the batch dimensions, the left free ones and the right free ones, in the order StableHLO gives them. `reduce` applies `add` (`ggml_sum_rows`) or `maximum` (`ggml_pool_1d`) over any set of dimensions, and its `init` must be the identity (0, or -inf). `maximum` and `minimum` of run-time floats, and `compare` of run-time floats or i1 values, run XLA CPU's semantics on the device (next section). `and`, `or`, `xor` and `not` take i1 values, known or run-time, and i32 values known at compile time. A hex literal is accepted as a scalar infinity, or as the quoted raw bytes MLIR prints for a large constant (`dense<"0x...">`, little-endian, one byte per i1). `rsqrt` is one over a square root. `slice` takes static starts, limits and strides (a stride on the last dimension is applied on a transposed copy, because a ggml view cannot step along its first dimension). `concatenate` takes any number of operands. `pad` takes edge padding with a constant zero and no interior padding. `sign` is a composite, because ggml-rd's `SGN` returns `-0` for a zero (ggml-cpu returns `+0`) and XLA returns the zero it was given: `s = sgn(x)`, `ss = s*s`, `c = s + sgn(1/x) * (1 - ss)`, `sign = c * ss`, which is exact for zeros of either sign, infinities and denormals (XLA flushes a denormal to a signed zero). A NaN is found as the one input where `ss` and `sgn(1/x)` are both 0, and the result is multiplied by `1 + 0/(1 - ind)`, which is NaN there (0/0) and 1 elsewhere.

`compare` of known values folds on the host (EQ, NE, LT, LE, GT, GE; SIGNED, UNSIGNED and FLOAT, not TOTALORDER). When all of `select`'s operands are known it folds. Otherwise each element is picked by a gather: both operands are laid out as rows of one element, the second after the first, and `ggml_get_rows` takes row `i` or row `N + i`. It is a 32-bit copy (`get_rows_b32`), so no element is rounded. A predicate computed at run time gives the row index on the device, `iota + N * (1 - p)` (exact up to 2^24 elements), converted to i32. XLA CPU on the reference desk (x86) returns a denormal operand of `select` as the zero of its sign (the Mac's XLA keeps it), so the gather's result is flushed with `xla_max(x, x)`. `mask_where`, `select_const_mask`, `causal_softmax_pinned` and `where_runtime` hold it.

## maximum, minimum and compare at run time

Measured on XLA CPU (jax 0.12.0.1) on the Zen 2 desk and an Apple M-series Mac, over every pair of 14 special values (zeros of both signs, infinities, NaN, denormals, the smallest normal, the largest float): XLA reads denormal inputs as zero of their sign (it runs with DAZ and FTZ), `maximum` and `minimum` return a NaN if either input is one and order -0 below +0, and `compare` is the IEEE comparison after that flush (`NE` true for a NaN). The model equals XLA on all 196 pairs except NaN payloads (x86 XLA ORs the inputs' bits for `minimum`), which the probe counts as equal. ggml has no op with these semantics, so `lean/Ggml/SlangCodegen/XlaCompare.lean` emits eight kernels (`max_xla_f32`, `min_xla_f32`, `cmp_{gt,ge,lt,le,eq,ne}_xla_f32`), with the flush done on the exponent bits so no driver's denormal mode matters. `guest/ggml-rd/ops/xla_compare.cpp` packs them as `GGML_OP_MAP_CUSTOM2` with a CPU reference. A run-time i1 is an f32 tensor of 0 and 1: `and` is the minimum, `or` the maximum, `xor` not-equal and `not` equal-to-zero. `relu_special`, `clip_special`, `maximum_pairs`, `minimum_pairs`, `compare_pairs` and `logic_runtime` are bit-equal on the RTX 4090; with the max and min kernels swapped, five of them fail. `multiply_denormal` shows that the RTX 4090 already flushes a denormal product to zero, as XLA does.

## Refused, by design

Everything else. The error is `UNIMPLEMENTED` for valid StableHLO outside the list (another op, another element type, i32 or i1 values computed at run time, a compare or select of i32 values computed at run time, more than four dimensions above size one, a dynamic dimension, a recursive call, a non-finite scalar hex constant other than an infinity, returning a parameter unchanged) and `INVALID_ARGUMENT` for text that is malformed or inconsistent (an undefined operand or function, a result type that does not match the op, a shape mismatch). The probe holds all 25 refusals.

## Bit equality with XLA CPU

The probe compares every case with the output XLA's CPU backend gave for the same JAX module. 66 of 74 cases are bit-equal on the RTX 4090 (and the first 58 cases, 51 bit-equal, on the RTX 3090 before). Eight are not, and each carries its measured reason in `tools/gen_stablehlo_cases.py` (`XFAIL`). They must meet the functional bar (relative L2 of 1e-4 for f32, 1e-3 for bf16) and fail the probe if one becomes bit-equal, so the list only shrinks.

| Case | Why it is not bit-equal yet |
|---|---|
| reduce_sum_random, mean_last | XLA CPU's float summation order is an irregular tree that changes with the shape, the signature of LLVM reassociating an unrolled chain of adds. `tools/reveal_sum_order.py` reveals it (the FPRev method): `(((5+(3+4))+(2+(0+1)))+((10+(9+(8+(6+7))))+((14+15)+(13+(11+12)))))` for 8 rows of 16, and a different tree for 32, 48 and 64 elements. No closed-form order reproduces it; only a table of revealed trees per shape, tied to this CPU and XLA build, could. XLA lowers the reduce to one `vector.multi_reduction` (xla/backends/cpu/codegen/tiled), so the order is chosen after XLA's own code. ggml-rd sums 256 interleaved lanes. |
| mlp_pre, mlp_f32, dot_batched | XLA CPU's dot (a YNNPACK fusion) accumulates in a shape-dependent tree. `tools/reveal_dot_order.py` shows that for 4x16 @ 16x8 it is the same tree as a float reduce of 16, and a sequential fused multiply-add chain matches 14 of 32 elements. The bias add after the dot is exact (dot then add equals XLA in 32 of 32), and so is the `exp` after it. |
| softmax_f32, rms_norm_f32 | each contains a float sum or rsqrt from the rows above. `softmax_pinned`, the same softmax with the sum order written out, is bit-equal. |
| causal_attention | JAX's `dot_product_attention(is_causal=True)`: its two dots and its softmax sum are the cases above (29 of 256 elements differ, by at most 8 ulp, rel-L2 2.7e-8). The mask, maximum, exp and divide around them are bit-equal (`causal_softmax_pinned`), and so are dots with two free or two batch dimensions whose partial sums are exact. |

`exp` and `log` are bit-equal. XLA CPU does not call libm for them: `polynomial_approximations.cc` (revision d867969800d4) rewrites them to a Cephes-style polynomial, and `lean/Ggml/SlangCodegen/ExactExpLog.lean` is that code, with the fused multiply-adds LLVM chose (found by comparing a host model with XLA over all 2^32 inputs: 0 differ) and XLA's flush of subnormals done with the exponent bits. `tests/exact_exp_log` holds the host model and the sweep. It changes the `exp_f32` and `log_f32` kernels, which ggml's EXP and LOG use, to XLA's polynomial: within 2^-22.5 relative of the true value, a few ulp from libm's. The `exp` inside softmax, SiLU and flash attention is unchanged. `exp_stress` and `log_stress` run 4096 inputs each (random bit patterns, both clamp edges, the subnormal boundary, zeros, subnormals, infinities and a NaN) and are bit-equal on the GPU; with the old shaders they differ in 2045 and 1465 of 4096.

`divide` by a broadcast is lowered as XLA does: its simplifier rewrites `a / broadcast(b)` as `a * broadcast(1 / b)`, which rounds twice. For a divisor of 3 the two forms differ in the last place, and with the rewrite `divide_by_const3`, `divide_by_row` and `softmax_pinned` are bit-equal.

`sine` and `cosine` are bit-equal to XLA CPU, which calls the C runtime's `sinf` and `cosf` (UCRT on the Windows desk): ggml's SIN and COS now run `ucrt_sinf` and `ucrt_cosf` (`lean/Ggml/SlangCodegen/ExactSinCos.lean`), a translation of the runtime's FMA path in double precision (Taylor polynomials, a two-constant reduction by pi/2, and Payne-Hanek with a 176-byte table for large arguments; needs Vulkan Float64 and Int64, which the RTX 3090 through Godot provides). `tests/xla_sincos` holds the translation in C, equal to the runtime's `sinf` and `cosf` for all 2^32 inputs, the disassembly notes, and the measurements. `sin_stress` and `cos_stress` run 4096 inputs each (every range up to 1e10, random bit patterns, the algorithm's thresholds, zeros, infinities, a NaN) on the GPU, bit-equal; the driver's own `sin` and `cos` differed from XLA in 3738 and 3430 of 4096. This is UCRT's algorithm, so it matches on Windows with FMA; another platform's libm gives other bits.

`rope_neox` (rotary embedding, rotate-halves layout, with the frequency table as an input) is bit-equal. A table computed in the program needs `power`, which is the C runtime's `powf` and is not lowered yet (see `tests/xla_sincos`), and `iota`.

`rsqrt` is bit-equal on the machine the reference was made on, an AMD Ryzen 7 3800X (Zen 2). XLA CPU's `rsqrt` (`xla/codegen/intrinsic/rsqrt.cc`) is the CPU's estimate instruction, `rsqrtps`, and one Newton step with both multiply-adds fused, and its source says it is not portable across CPUs: Intel has another table and two steps. ggml has no `rsqrt`, so the lowering emits `GGML_OP_MAP_CUSTOM1` with `ggml_rd_rsqrt_zen2`, which is also the CPU reference: the estimate from a 2 x 4096-entry table, the way Bochs emulates Intel's (`guest/ggml-rd/rsqrt_zen2.h`). ggml-rd recognises that function and runs the kernel `rsqrt_zen2_f32` (`lean/Ggml/SlangCodegen/ExactRsqrt.lean`): the same table as the exact function plus a 2-bit residual per entry, since a Slang function cannot hold an array. `tests/xla_rsqrt` holds the checks: the table equals the instruction for all 2^32 inputs, the kernel's arithmetic equals the hardware model of XLA for all 2^32 inputs, and that model equals `jax.lax.rsqrt` for all 2^32. `rsqrt_stress` runs 4096 inputs (random bit patterns, zeros, subnormals, infinities, a NaN) on the GPU, bit-equal; pointed at the wrong kernel it differs in 3153 of 4096. On another CPU XLA's `rsqrt` differs from this by an ulp or two.

Integer-valued sums, `maximum` reductions (single and multi-dimension), bf16 sums and sums over a non-last or non-adjacent dimension are bit-equal.

## Pinning a sum's order

StableHLO leaves the order of a `reduce` unspecified, and XLA CPU's is an arbitrary tree (below). A program that needs a defined order has to write it, and write it so XLA keeps it. A chain of size-2 reduces does not hold: XLA's simplifier merges it into one `reduce-window` and the order changes. Explicit adds of slices do. `tree_sum` in `tools/gen_stablehlo_cases.py` adds lane `i` to lane `i + n/2` and halves until one is left, and it equals XLA CPU in every row for every shape tried (16 to 1024 elements, 1 to 64 rows). `pinned_sum_tree`, `pinned_mean` and `pinned_rms_scale` use it and are bit-equal; this is how to get a bit-equal float sum.

XLA also rewrites `a / sqrt(v)` into `a * rsqrt(v)`, so a full RMS norm meets the approximate `rsqrt` again; `rms_norm_f32` stays on the list below.

## Coverage

The goal is the whole of StableHLO, and every graph the ggml-rd probes run today (qwen, dit, sconv, the two Kimodo layers). Today it is 33 ops plus calls. RoPE with its frequency table as an input and JAX's causal attention compile and run; a computed frequency table needs `power`, and attention's dots and sums need XLA's summation order to compare bit for bit. Still missing for those graphs: `tanh` and `erf` (GELU), `gather` (an embedding lookup, whose index arithmetic is i32 at run time), `convolution`, and run-time i32 arithmetic.

## Regenerating the cases

```sh
pixi run --manifest-path <stablehlo-gen>/pixi.toml python tools/gen_stablehlo_cases.py guest/ggml_test/stablehlo_cases.inc
```

The environment is Python 3.12 with JAX from PyPI (conda-forge has no `jaxlib` for Windows). The header of the generated file names the JAX version.

## XLA CPU's dot order, measured (tools/dot_trees.py, jax 0.12.0.1)

The summation tree of `a @ w` over the contracted dimension K, found with FPRev for M x K @ K x N:

- With N > 1 the tree depends on K alone: K=16 gives the same tree for M,N = 1x8, 4x8, 8x8, 16x16 and 3x5. With N = 1 it is another tree,
  a lane-wise one that pairs element i with i+8 (K=16: ((0+8)+(1+9))+... ), the matrix-vector path.
- For N > 1 the trees are chains inside blocks, joined unevenly. K=2..4: a left-to-right chain, (3+(2+(0+1))). K=5: ((3+4)+(2+(0+1))).
  K=9: (2+(0+1)) + ((8+(6+7)) + (5+(3+4))), three chains of three. K=12: a chain of six over 6..11 added to the chains over 0..5.
  K=16: blocks 0..5, 6..10 and 11..15. K=24 and K=32 are bigger and uneven. No closed rule is found yet.

So a bit-equal dot needs XLA's tree per K (a table, or a rule recovered from larger K), expressed as gathers and adds. ggml's mul_mat accumulates in its own order.
