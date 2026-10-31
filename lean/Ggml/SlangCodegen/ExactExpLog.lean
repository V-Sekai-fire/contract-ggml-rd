import LeanSlang
import Ggml.SlangCodegen.Common

/-!
# `Ggml.SlangCodegen.ExactExpLog` — XLA CPU's f32 `exp` and `log`, bit for bit

XLA CPU does not call libm for f32 `exp` and `log`: `xla/backends/cpu/codegen/polynomial_approximations.cc`
(`GenerateVF32Exp`, `GenerateVF32Log`, XLA revision d867969800d4, the one JAX 0.12.0.1 pins) rewrites
them to a Cephes-style polynomial. The functions here are that code, so `exp_f32` and `log_f32` give the
bits XLA CPU gives:

- `exp_xla(x)`: `x` clamped to [-87.8, 88.8], `n = floor(fma(x, log2e, 0.5))` clamped to [-127, 128],
  `a = x - n ln2` in two Cody-Waite steps, a degree-6 polynomial `1 + a + a^2 p(a)`, and `2^n` formed
  from the exponent field (with the last factor of two split off so `n = 128` does not overflow).
- `log_xla(x)`: the exponent split off the bits, the mantissa moved to [sqrt(1/2), sqrt(2)), a degree-9
  polynomial in `x - 1`, and the `n ln2` terms added back in two parts.

LLVM contracted XLA's multiply-adds when it compiled them. Which products it fused was found by
comparing a host model with XLA CPU over all 2^32 inputs: every `MulAdd` is one fused multiply-add,
`x - c n` is one in the exponent reduction, and the last step of `log` is `fma(y, x3, q1 e)`. Both host
models equal XLA CPU for every one of the 2^32 bit patterns (NaN payloads aside, which IEEE 754 leaves
open). XLA CPU also flushes subnormals: a subnormal input is read as a signed zero and a subnormal
result is written as one, which is done here with the exponent bits, so no driver setting matters.
Every statement is `precise` (SPIR-V NoContraction), so the GPU contracts nothing beyond the explicit
`fma`.
-/

namespace Ggml.SlangCodegen.ExactExpLog

open LeanSlang
open Ggml.SlangCodegen.Common

private def f32 : SlangType := .scalar .float
private def u32 : SlangType := .scalar .uint
private def i32 : SlangType := .scalar .int

private def fparam (n : String) : SlangBinding :=
  ⟨n, f32, Semantic.none, none, none, .qIn⟩

private def pf (name : String) (init : SlangExpr) : SlangStmt := .declarePrecise f32 name (some init)
private def fl (x : Float) : SlangExpr := .litFloatExact x
private def fma (a b c : SlangExpr) : SlangExpr := .call "fma" [a, b, c]
private def neg (e : SlangExpr) : SlangExpr := .un "-" e
private def sub (a b : SlangExpr) : SlangExpr := .bin "-" a b
private def band (a b : SlangExpr) : SlangExpr := .bin "&" a b
private def bor (a b : SlangExpr) : SlangExpr := .bin "|" a b
private def eq (a b : SlangExpr) : SlangExpr := .bin "==" a b
private def lt (a b : SlangExpr) : SlangExpr := .bin "<" a b
private def gt (a b : SlangExpr) : SlangExpr := .bin ">" a b
private def asuint (e : SlangExpr) : SlangExpr := .call "asuint" [e]
private def asfloat (e : SlangExpr) : SlangExpr := .call "asfloat" [e]
private def isnan (e : SlangExpr) : SlangExpr := .call "isnan" [e]

/-- `x` with a subnormal read as a zero of the same sign (XLA CPU runs with denormals-are-zero). -/
private def daz : List SlangStmt :=
  [ .declare u32 "xb" (some (asuint (v "x")))
  , pf "xi" (.ternary (eq (band (v "xb") (u 0x7f800000)) (u 0)) (asfloat (band (v "xb") (u 0x80000000))) (v "x")) ]

/-- `exp_xla`: see the module doc. -/
def fnExpXla : SlangFunctionDecl :=
  { retType := f32
  , name := "exp_xla"
  , params := [fparam "x"]
  , body :=
      [ .ifThen (isnan (v "x")) [ .ret (some (v "x")) ] [] ]
    ++ daz ++
      [ pf "c0" (.ternary (lt (v "xi") (fl (-87.8))) (fl (-87.8)) (v "xi"))
      , pf "c1" (.ternary (gt (v "c0") (fl 88.8)) (fl 88.8) (v "c0"))
      , pf "n0" (.call "floor" [fma (v "c1") (fl 1.44269504088896341) (fl 0.5)])
      , pf "n1" (.ternary (lt (v "n0") (fl (-127.0))) (fl (-127.0)) (v "n0"))
      , pf "n" (.ternary (gt (v "n1") (fl 128.0)) (fl 128.0) (v "n1"))
      , pf "a0" (fma (neg (fl 0.693145751953125)) (v "n") (v "c1"))
      , pf "a" (fma (neg (fl 1.4286067653301877e-6)) (v "n") (v "a0"))
      , pf "z0" (fma (v "a") (fl 1.9875691500e-4) (fl 1.3981999507e-3))
      , pf "z1" (fma (v "z0") (v "a") (fl 8.3334519073e-3))
      , pf "z2" (fma (v "z1") (v "a") (fl 4.1665795894e-2))
      , pf "z3" (fma (v "z2") (v "a") (fl 1.6666665459e-1))
      , pf "z4" (fma (v "z3") (v "a") (fl 5.0000001201e-1))
      , pf "aa" (mul (v "a") (v "a"))
      , pf "z5" (fma (v "z4") (v "aa") (v "a"))
      , pf "z" (add (fl 1.0) (v "z5"))
      , .declare u32 "shl" (some (.bin "<<" (.cast u32 (add (.cast i32 (v "n")) (.litInt 127))) (u 23)))
      , .declare u32 "p2" (some (.call "min" [v "shl", u 0x7f000000]))
      , .declare u32 "remb" (some (sub (v "shl") (v "p2")))
      , pf "zs" (asfloat (add (asuint (v "z")) (v "remb")))
      , pf "r" (mul (v "zs") (asfloat (v "p2")))
      , .ret (some (.ternary (eq (band (asuint (v "r")) (u 0x7f800000)) (u 0)) (asfloat (band (asuint (v "r")) (u 0x80000000))) (v "r"))) ] }

/-- `log_xla`: see the module doc. -/
def fnLogXla : SlangFunctionDecl :=
  { retType := f32
  , name := "log_xla"
  , params := [fparam "x"]
  , body :=
      [ .ifThen (isnan (v "x")) [ .ret (some (v "x")) ] [] ]
    ++ daz ++
      [ .ifThen (eq (v "xi") (fl 0.0)) [ .ret (some (asfloat (u 0xff800000))) ] []
      , .ifThen (lt (v "xi") (fl 0.0)) [ .ret (some (asfloat (u 0xffffffff))) ] []
      , .ifThen (eq (asuint (v "xi")) (u 0x7f800000)) [ .ret (some (v "xi")) ] []
      , .declare u32 "bits" (some (asuint (v "xi")))
      , .declare i32 "emm" (some (sub (.cast i32 (.bin ">>" (v "bits") (u 23))) (.litInt 127)))
      , pf "m0" (asfloat (bor (band (v "bits") (u 0x807fffff)) (u 0x3f000000)))
      , pf "e0" (add (fl 1.0) (.cast f32 (v "emm")))
      , .declare (.scalar .bool) "low" (some (lt (v "m0") (fl 0.707106781186547524)))
      , pf "t1" (.ternary (v "low") (v "m0") (fl 0.0))
      , pf "m1" (sub (v "m0") (fl 1.0))
      , pf "e" (sub (v "e0") (.ternary (v "low") (fl 1.0) (fl 0.0)))
      , pf "m" (add (v "m1") (v "t1"))
      , pf "x2" (mul (v "m") (v "m"))
      , pf "x3" (mul (v "x2") (v "m"))
      , pf "y0" (fma (v "m") (fl 7.0376836292e-2) (fl (-1.1514610310e-1)))
      , pf "y1" (fma (v "m") (fl (-1.2420140846e-1)) (fl 1.4249322787e-1))
      , pf "y2" (fma (v "m") (fl 2.0000714765e-1) (fl (-2.4999993993e-1)))
      , pf "y3" (fma (v "y0") (v "m") (fl 1.1676998740e-1))
      , pf "y4" (fma (v "y1") (v "m") (fl (-1.6668057665e-1)))
      , pf "y5" (fma (v "y2") (v "m") (fl 3.3333331174e-1))
      , pf "y6" (fma (v "y3") (v "x3") (v "y4"))
      , pf "y7" (fma (v "y6") (v "x3") (v "y5"))
      , pf "q1e" (mul (fl (-2.12194440e-4)) (v "e"))
      , pf "y8" (fma (v "y7") (v "x3") (v "q1e"))
      , pf "hx2" (mul (fl 0.5) (v "x2"))
      , pf "r0" (sub (v "m") (v "hx2"))
      , pf "r1" (add (v "r0") (v "y8"))
      , pf "q2e" (mul (fl 0.693359375) (v "e"))
      , .ret (some (add (v "r1") (v "q2e"))) ] }

end Ggml.SlangCodegen.ExactExpLog
