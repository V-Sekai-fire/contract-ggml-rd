# XLA CPU sin and cos

Measured on an AMD Ryzen 7 3800X, Windows, jaxlib 0.12.0.1 (XLA at d867969800d4).

    python sincos_dump.py        # 999,246 f32 inputs (uniform in +-10, +-1000, +-1, and random bit patterns) and XLA's sin and cos of them
    clang -O1 -ffp-contract=off sincos_libm.c && ./a.out
    of 999246 inputs differ from XLA: sinf 0 | (float)sin(double) 1319 | cosf 0 | (float)cos(double) 1320

- XLA CPU's sin and cos are the C runtime's sinf and cosf, bit for bit (UCRT here). XLA's source has no f32 sin kernel;
  the HLO is a plain `sine`, and the lowering is a libm call.
- They are not the correctly rounded value: rounding the double sin to f32 differs in 1,319 (0.13%).
- XNNPACK's rational 5/4 sin and cos (f32-vsin), which YNNPACK uses on its own path, differ from XLA in 58% of inputs, with or
  without fused multiply-adds (an earlier sincos_model.c; removed), so XLA's jitted sine does not go through it.
- The driver's sin and cos on the GPU were 11 to 21 ulp off XLA (3738 and 3430 of 4096 differ in sin_stress and cos_stress).

## UCRT's sinf and cosf, translated (the bit-equal kernel)

ucrtbase.dll (Windows 10/11, x64) dispatches on the CPU: with FMA it runs a double-precision algorithm, read from the disassembly
(`llvm-objdump -d`, sinf at RVA 0xAC360, cosf at 0xA7F70) and its constants (ucrt_consts.py reads them from the DLL):

- |x| < 2^-13: x itself (sin), 1.0 (cos); below 2^-7 a cubic (1 - x*(x*0.5) for cos); to pi/4 a Taylor polynomial in double with fused
  multiply-adds (1/3!, 1/5!, 1/7!, 1/9! for sin; 1/2!, 1/4!, 1/6!, 1/8!, 1/10! for cos);
- above pi/4: n = trunc(|x| * 2/pi + 0.5), r = (|x| - n*1.5707963267341256) - n*6.077100506506192e-11, the polynomial for n odd/even, the
  sign from n & 3 (and from x for sin); this holds to 0x4b800456 (sinf) and 0x4f490fdb (cosf);
- beyond: Payne-Hanek with a 176-byte table of 2/pi at RVA 0x107360, a 53 x 192-bit product, rounding to the nearest quadrant, and for
  cosf a double-double product with pi/2 (1.5707963267948966 split with 1.5707963109016418, 1.5893254712295857e-08, 6.123233995736765e-17).

ucrt_libm.c is that translation; ucrt_libm_sweep.c compares it with the runtime's sinf and cosf for all 2^32 inputs:

    clang -O2 -mavx2 -mfma -ffp-contract=off ucrt_libm_sweep.c && a.exe
    ucrt_sinf vs sinf over 2^32 inputs: 0 differ (0 with |x| >= 2^24)
    ucrt_cosf vs cosf over 2^32 inputs: 0 differ (0 with |x| >= 2^24)

lean/Ggml/SlangCodegen/ExactSinCos.lean is the same algorithm as a Slang kernel (double, uint64_t), used for ggml's SIN and COS.
Windows only: another platform's libm gives other bits, and a CPU without FMA takes UCRT's other path.

## power

XLA CPU's `power` is the runtime's `powf` too: pow_dump.py writes 600,000 (a, b) pairs and XLA's `jnp.power`, pow_libm.c compares:

    of 600000 inputs differ from XLA: powf 0 | (float)pow(double) 53 | expf(b*logf(a)) 393319

So a bit-equal `power` needs UCRT's powf (FMA path at ucrtbase.dll RVA 0xA9F20, branch at +0x7b6, with log and exp tables); not translated yet.
A rotary embedding whose frequency table is an input (`rope_neox`) needs only sin, cos and the ops already lowered, and is bit-equal.
