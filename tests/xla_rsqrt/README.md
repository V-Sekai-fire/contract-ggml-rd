Host model of XLA CPU's f32 rsqrt (xla/codegen/intrinsic/rsqrt.cc, XLA revision d867969800d4).

    clang -O2 -ffp-contract=off -mavx2 -mfma -shared -o xla_rsqrt.dll xla_rsqrt.c
    python sweep_vs_xla.py      # all 2^32 inputs against jax.lax.rsqrt on CPU

Measured on an AMD Ryzen 7 3800X (Zen 2, no AVX-512), jax 0.12.0.1: 0 inputs differ. The model is the
hardware estimate (_mm_rsqrt_ss, equal to the 8-wide rsqrtps here) and one Newton step with both
multiply-adds fused: r = fma(-0.5 r, fma(x r, r, -1), r). rsqrt.cc uses one step on AMD and two on
Intel (more accurate AMD estimate), and the estimate itself is a table that differs by vendor.
estimate_table.py shows the estimate depends only on the top 12 mantissa bits and the exponent parity
(2 x 4096 entries, 12-bit mantissa), with no closed form.

What was tried to turn the estimate table into a formula (all measured on the 2 x 4096 entries, in units of 2^-13):

- Exact 1/sqrt at the group's low, middle or high edge, floored or rounded: 1,100 to 3,800 of 4,096 entries wrong.
- Best single sample point and offset: 958 to 969 wrong; the entries differ from the smooth curve by -1, 0 or +1.
- Polynomials of degree 1 to 3 under a floor, over 1 to 64 segments: none fits a segment of 16 or more, whatever the degree.
- Linear interpolation between nodes (exact or rounded, any quantisation, groups of 4 to 32): 779 to 1,100 wrong; a continuous chain of real-valued nodes is infeasible (floor_of_line.py).
- Tangent lines of the exact function: about 980 wrong at every block size.

What holds: every run of 8 consecutive entries is the floor of a straight line (floor_of_line.py: 512 of 512 for G=8), the 512 runs have only 18 (127) and 15 (128) distinct shapes (block_shapes.py), and the offsets of runs drift smoothly with a +-1 toggle on bit 3 of the index. That is a ROM of about 512 (base, slope) pairs per binade, not an arithmetic formula.
