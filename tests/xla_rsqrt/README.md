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

A witness search with V-Sekai-fire/plausible-witness-dag (witness/, `lake exe rsqrt_witness`): hypotheses are indices
into three families (floor of node interpolation, of the tangent line, of the exact function on a coarser index grid),
and the predicate is "reproduces all 2 x 4096 entries". A planted control (a table made by index 3000) is found at
exactly 3000. On the real table the ladder ends `provablyNone` over all 4096 hypotheses; the closest, index 2627, is
wrong in 1,959 of 8,192 entries. table_dag.py hash-conses the table into a decision DAG with relative values:
splitting the index from its high bit gives 186 distinct nodes (binade 127) and 167 (binade 128), where the low bit
first gives 2,368 and 2,059. That DAG is the compact form of the ROM, about 350 nodes in all.

Web search for a closed form (2026-10-31). Search engines (Bing, DuckDuckGo, Mojeek) refuse scripted queries; Stack Exchange
and Sourcegraph answered. Nothing publishes AMD's table or rule. What exists:

- Bochs (cpu/sse_rcp.cc, Stanislav Shwartsman, LGPL) emulates Intel's rcpps and rsqrtps with a 2048-entry rcp table and two
  1024-entry rsqrt tables, and states the rule: the result is 1/(1.yyyyyyyyyy1) rounded to nearest, the index plus a trailing 1,
  i.e. the function at the interval midpoint. intel_closed_form.py checks it against Bochs's tables: round(1/sqrt(midpoint)) to
  13 bits gives 0 mismatches of 1024 in both tables (11 and 12 bits: 764 and 502). Intel's rsqrtps therefore has a closed form.
- AMD's is a 4096-entry table (12-bit index), not Intel's 1024. The same rule at 12 bits leaves about 1,100 of 4,096 wrong
  (see above), and amd_from_intel.py (Intel-style midpoint nodes interpolated four ways) leaves 2,118 of 8,192 wrong.
- Stack Overflow 13102167 confirms the two vendors return different rcpps and rsqrtps values.
