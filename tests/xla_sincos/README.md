# XLA CPU sin and cos

Measured on an AMD Ryzen 7 3800X, Windows, jaxlib 0.12.0.1 (XLA at d867969800d4).

    python sincos_dump.py        # 999,246 f32 inputs (uniform in +-10, +-1000, +-1, and random bit patterns) and XLA's sin and cos of them
    clang -O1 -ffp-contract=off sincos_libm.c && ./a.out
    of 999246 inputs differ from XLA: sinf 0 | (float)sin(double) 1319 | cosf 0 | (float)cos(double) 1320

- XLA CPU's sin and cos are the C runtime's sinf and cosf, bit for bit (UCRT here). XLA's source has no f32 sin kernel;
  the HLO is a plain `sine`, and the lowering is a libm call.
- They are not the correctly rounded value: rounding the double sin to f32 differs in 1,319 (0.13%).
- sincos_model.c is XNNPACK's rational 5/4 sin/cos (f32-vsin), which YNNPACK uses on its own path. It does not match XLA
  (58% differ, with or without fused multiply-adds), so XLA's jitted sine does not go through it.
- The GPU kernel (the driver's sin) is 11 to 21 ulp off XLA; 3738 of 4096 and 3430 of 4096 differ in sin_stress and cos_stress.
- So the target is UCRT's sinf, and another platform's libm gives other bits. Matching it needs that algorithm (its
  reduction and polynomial) recovered from the library.
