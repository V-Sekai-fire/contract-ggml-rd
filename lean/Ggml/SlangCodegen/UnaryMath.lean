import LeanSlang
import Ggml.SlangCodegen.Common
import Ggml.SlangCodegen.Unary
import Ggml.SlangCodegen.ExactExpLog
import Ggml.SlangCodegen.ExactRsqrt

/-!
# `Ggml.SlangCodegen.UnaryMath` — family K1: the f32 math unaries

ABS, SGN, STEP, TANH, EXP, FLOOR (`GGML_OP_UNARY`) and SQR, LOG, SIN, COS
(ops of their own in ggml), the element-wise ops interactor-nx-ggml's
graph builder and interactor-rf-detr-ggml's decoder, deformable attention
and loss build from. Each is `Unary.shader` (one thread per destination
element, `dst[off(dst, i)] = f(s0[off(s0, i)])`, both sides strided, in
place safe) with one function; the packer is
`guest/ggml-rd/ops/unary_math.cpp`.

## The functions, against ggml-cpu (`src/ggml-cpu/unary-ops.cpp`)

ggml-cpu computes every one of these in f32 (`op_abs` .. `op_floor`,
through `unary_op<op>` with `float` source and destination): none reads an
f16 table (`GGML_GELU_FP16` and `GGML_GELU_QUICK_FP16` are the only table
macros in `vec.h`, for GELU and GELU_QUICK), so the f32 formula is the
whole reference:

- `abs(x) = fabsf(x)`; `sqr(x) = x * x`; `floor(x) = floorf(x)`;
  `exp(x) = expf(x)`; `log(x) = logf(x)`; `sin(x) = sinf(x)`;
  `cos(x) = cosf(x)`. Slang's `abs`, `floor`, `exp`, `log`, `sin`, `cos`
  compile to `fabsf`, `floorf`, `expf`, `logf`, `sinf`, `cosf` on the cpp
  target (the prelude's `F32_*`), so the host emit is bit-exact against
  ggml-cpu; on the GPU `exp`, `log`, `sin`, `cos` are the driver's, within
  test-backend-ops' NMSE.
- `sgn(x) = (x > 0) ? 1 : ((x < 0) ? -1 : 0)`, `step(x) = (x > 0) ? 1 : 0`:
  ggml-cpu's ternaries, so a NaN gives 0 on both.
- `tanh(x) = tanhf(x)`, with the argument clamped to [-10, 10] by two
  ternaries (a NaN passes through, unlike Slang's `clamp`): tanhf is
  exactly 1.0f from 9.01 on, so the clamp changes no result, and it keeps a
  driver whose tanh is `(e^2y - 1) / (e^2y + 1)` from returning inf/inf at
  |x| ~ 150 (test-backend-ops draws unary inputs from [-150, 150]).
-/

namespace Ggml.SlangCodegen.UnaryMath

open LeanSlang
open Ggml.SlangCodegen.Common
open Ggml.SlangCodegen.Unary
open Ggml.SlangCodegen.ExactExpLog
open Ggml.SlangCodegen.ExactRsqrt

private def floatParam (n : String) : SlangBinding :=
  ⟨n, .scalar .float, Semantic.none, none, none, .qIn⟩
private def float1 (name : String) (body : List SlangStmt) : SlangFunctionDecl :=
  { retType := .scalar .float, name := name, params := [floatParam "x"], body := body }

/-- `float sgn(float x) { return (x > 0) ? 1 : ((x < 0) ? -1 : 0); }` -/
def fnSgn : SlangFunctionDecl :=
  float1 "sgn"
    [ .ret (some (.ternary (.bin ">" (v "x") (fl 0.0)) (fl 1.0)
        (.ternary (.bin "<" (v "x") (fl 0.0)) (fl (-1.0)) (fl 0.0)))) ]

/-- `float step(float x) { return (x > 0) ? 1 : 0; }` -/
def fnStep : SlangFunctionDecl :=
  float1 "step_ggml" [ .ret (some (.ternary (.bin ">" (v "x") (fl 0.0)) (fl 1.0) (fl 0.0))) ]

/-- `float tanh_clamped(float x)`: tanh of x clamped to [-10, 10] (see the
    module doc); NaN stays NaN. -/
def fnTanhClamped : SlangFunctionDecl :=
  float1 "tanh_clamped"
    [ .declInit (.scalar .float) "y"
        (.ternary (.bin ">" (v "x") (fl tanhClamp)) (fl tanhClamp)
          (.ternary (.bin "<" (v "x") (fl (-tanhClamp))) (fl (-tanhClamp)) (v "x")))
    , .ret (some (.call "tanh" [v "y"])) ]

def absF32 : SlangShaderModule := shader [] (.call "abs" [v "x"])
def sgnF32 : SlangShaderModule := shader [fnSgn] (.call "sgn" [v "x"])
def stepF32 : SlangShaderModule := shader [fnStep] (.call "step_ggml" [v "x"])
def expF32 : SlangShaderModule := shader [fnExpXla] (.call "exp_xla" [v "x"])
def tanhF32 : SlangShaderModule := shader [fnTanhClamped] (.call "tanh_clamped" [v "x"])
def floorF32 : SlangShaderModule := shader [] (.call "floor" [v "x"])
def sqrF32 : SlangShaderModule := shader [] (mul (v "x") (v "x"))
def logF32 : SlangShaderModule := shader [fnLogXla] (.call "log_xla" [v "x"])
def rsqrtZen2F32 : SlangShaderModule := shader rsqrtZen2Helpers (.call "rsqrt_zen2" [v "x"])
def sinF32 : SlangShaderModule := shader [] (.call "sin" [v "x"])
def cosF32 : SlangShaderModule := shader [] (.call "cos" [v "x"])

/-- The kernels this module contributes, by their kernels.txt names. -/
def kernels : List (String × SlangShaderModule) :=
  [ ("abs_f32", absF32)
  , ("sgn_f32", sgnF32)
  , ("step_f32", stepF32)
  , ("exp_f32", expF32)
  , ("tanh_f32", tanhF32)
  , ("floor_f32", floorF32)
  , ("sqr_f32", sqrF32)
  , ("log_f32", logF32)
  , ("sin_f32", sinF32)
  , ("cos_f32", cosF32)
  , ("rsqrt_zen2_f32", rsqrtZen2F32) ]

/-! ## Pins: each kernel is `Unary.sibling` with its helpers and its result. -/

example : LeanSlang.emit absF32 = sibling "" "dst[d] = abs(x);" := by native_decide
example : LeanSlang.emit expF32 = sibling "float exp_xla(float x) {
  if (isnan(x)) {
    return x;
  }
  uint xb = asuint(x);
  precise float xi = (((xb & 2139095040u) == 0u) ? asfloat((xb & 2147483648u)) : x);
  precise float c0 = ((xi < (-87.8f)) ? (-87.8f) : xi);
  precise float c1 = ((c0 > 88.8f) ? 88.8f : c0);
  precise float n0 = floor(fma(c1, 1.442695f, 0.5f));
  precise float n1 = ((n0 < (-127.0f)) ? (-127.0f) : n0);
  precise float n = ((n1 > 128.0f) ? 128.0f : n1);
  precise float a0 = fma((-0.69314575f), n, c1);
  precise float a = fma((-1.4286068e-6f), n, a0);
  precise float z0 = fma(a, 0.00019875691f, 0.0013981999f);
  precise float z1 = fma(z0, a, 0.008333452f);
  precise float z2 = fma(z1, a, 0.041665796f);
  precise float z3 = fma(z2, a, 0.16666666f);
  precise float z4 = fma(z3, a, 0.5f);
  precise float aa = (a * a);
  precise float z5 = fma(z4, aa, a);
  precise float z = (1.0f + z5);
  uint shl = (uint((int(n) + 127)) << 23u);
  uint p2 = min(shl, 2130706432u);
  uint remb = (shl - p2);
  precise float zs = asfloat((asuint(z) + remb));
  precise float r = (zs * asfloat(p2));
  return (((asuint(r) & 2139095040u) == 0u) ? asfloat((asuint(r) & 2147483648u)) : r);
}
\n" "dst[d] = exp_xla(x);" := by native_decide

example : LeanSlang.emit logF32 = sibling "float log_xla(float x) {
  if (isnan(x)) {
    return x;
  }
  uint xb = asuint(x);
  precise float xi = (((xb & 2139095040u) == 0u) ? asfloat((xb & 2147483648u)) : x);
  if ((xi == 0.0f)) {
    return asfloat(4286578688u);
  }
  if ((xi < 0.0f)) {
    return asfloat(4294967295u);
  }
  if ((asuint(xi) == 2139095040u)) {
    return xi;
  }
  uint bits = asuint(xi);
  int emm = (int((bits >> 23u)) - 127);
  precise float m0 = asfloat(((bits & 2155872255u) | 1056964608u));
  precise float e0 = (1.0f + float(emm));
  bool low = (m0 < 0.70710677f);
  precise float t1 = (low ? m0 : 0.0f);
  precise float m1 = (m0 - 1.0f);
  precise float e = (e0 - (low ? 1.0f : 0.0f));
  precise float m = (m1 + t1);
  precise float x2 = (m * m);
  precise float x3 = (x2 * m);
  precise float y0 = fma(m, 0.070376836f, (-0.1151461f));
  precise float y1 = fma(m, (-0.12420141f), 0.14249323f);
  precise float y2 = fma(m, 0.20000714f, (-0.24999994f));
  precise float y3 = fma(y0, m, 0.116769984f);
  precise float y4 = fma(y1, m, (-0.16668057f));
  precise float y5 = fma(y2, m, 0.3333333f);
  precise float y6 = fma(y3, x3, y4);
  precise float y7 = fma(y6, x3, y5);
  precise float q1e = ((-0.00021219444f) * e);
  precise float y8 = fma(y7, x3, q1e);
  precise float hx2 = (0.5f * x2);
  precise float r0 = (m - hx2);
  precise float r1 = (r0 + y8);
  precise float q2e = (0.6933594f * e);
  return (r1 + q2e);
}
\n" "dst[d] = log_xla(x);" := by native_decide

/-- The rsqrt_zen2 kernel is too long to pin as text (a 512-leaf `if` tree): pin its shape instead. -/
example : ((LeanSlang.emit rsqrtZen2F32).splitOn "if ((w < ").length = 512
    ∧ ((LeanSlang.emit rsqrtZen2F32).splitOn "dst[d] = rsqrt_zen2(x);").length = 2 := by native_decide

example : LeanSlang.emit floorF32 = sibling "" "dst[d] = floor(x);" := by native_decide
example : LeanSlang.emit sqrF32 = sibling "" "dst[d] = (x * x);" := by native_decide
example : LeanSlang.emit sinF32 = sibling "" "dst[d] = sin(x);" := by native_decide
example : LeanSlang.emit cosF32 = sibling "" "dst[d] = cos(x);" := by native_decide

example : LeanSlang.emit sgnF32 = sibling "float sgn(float x) {
  return ((x > 0.0f) ? 1.0f : ((x < 0.0f) ? (-1.0f) : 0.0f));
}

" "dst[d] = sgn(x);" := by native_decide

example : LeanSlang.emit stepF32 = sibling "float step_ggml(float x) {
  return ((x > 0.0f) ? 1.0f : 0.0f);
}

" "dst[d] = step_ggml(x);" := by native_decide

example : LeanSlang.emit tanhF32 = sibling "float tanh_clamped(float x) {
  float y = ((x > 10.0f) ? 10.0f : ((x < (-10.0f)) ? (-10.0f) : x));
  return tanh(y);
}

" "dst[d] = tanh_clamped(x);" := by native_decide

end Ggml.SlangCodegen.UnaryMath
