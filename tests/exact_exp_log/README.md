Host model of XLA CPU's f32 exp and log (polynomial_approximations.cc, XLA revision d867969800d4).

    clang -O2 -ffp-contract=off -mfma -shared -o xla_expl.dll xla_expl.c
    python sweep_vs_xla.py exp 3        # all 2^32 inputs against jax.numpy.exp on CPU; fuse mask 3
    python sweep_vs_xla.py log 17,49    # the same for log

Result with jax 0.12.0.1: 0 inputs differ for exp (fuse 3) and for log (fuse 17 and 49), NaN payloads aside.
`fuse` picks which multiply-add pairs are one fmaf; the Lean module ExactExpLog.lean uses the winning set.
