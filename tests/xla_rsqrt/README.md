Host model of XLA CPU's f32 rsqrt (xla/codegen/intrinsic/rsqrt.cc, XLA revision d867969800d4).

    clang -O2 -ffp-contract=off -mavx2 -mfma -shared -o xla_rsqrt.dll xla_rsqrt.c
    python sweep_vs_xla.py      # all 2^32 inputs against jax.lax.rsqrt on CPU

Measured on an AMD Ryzen 7 3800X (Zen 2, no AVX-512), jax 0.12.0.1: 0 inputs differ. The model is the
hardware estimate (_mm_rsqrt_ss, equal to the 8-wide rsqrtps here) and one Newton step with both
multiply-adds fused: r = fma(-0.5 r, fma(x r, r, -1), r). rsqrt.cc uses one step on AMD and two on
Intel (more accurate AMD estimate), and the estimate itself is a table that differs by vendor.
estimate_table.py shows the estimate depends only on the top 12 mantissa bits and the exponent parity
(2 x 4096 entries, 12-bit mantissa), with no closed form.
