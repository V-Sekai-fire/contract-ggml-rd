import LeanSlang
import Ggml.SlangCodegen.Common
import Ggml.SlangCodegen.Unary
import Ggml.SlangCodegen.ExactDivSqrt

/-!
# `Ggml.SlangCodegen.UnarySeeThrough` — SQRT and GELU_QUICK, f32 (family K1)

Two more element-wise kernels in `Unary`'s shape (one thread per
destination element, `dst[off(dst, i)] = f(s0[off(s0, i)])`, both sides
strided, no barriers, so the cpp emit runs the same module), the ones
the see-through engine (weftspun/see-through-cpp, an SDXL-class
layer-decomposition pipeline) needs beyond `Unary`. A module of their
own so a family added in parallel does not touch the same lines.

- `sqrt(x)`: `GGML_OP_SQRT` (a top-level op in this ggml, not a
  `ggml_unary_op`), ggml-cpu's `ggml_vec_sqrt_f32` = `sqrtf(x)`; one
  IEEE square root. A Vulkan `sqrt` may be 2.5 ULP off, so the kernel calls
  `sqrt_exact` (`ExactDivSqrt`), correctly rounded on both targets.
- `gelu_quick(x) = x * (1 / (1 + exp(-1.702 x)))`: `GGML_UNARY_OP_GELU_QUICK`,
  written as ggml-cpu's `ggml_gelu_quick_f32`,
  `x*(1.0f/(1.0f+expf(GELU_QUICK_COEF*x)))` with `GELU_QUICK_COEF = -1.702f`,
  the f32 formula `ggml_vec_gelu_quick_f32` runs when `GGML_GELU_QUICK_FP16`
  is not defined. ggml-cpu as vendored *does* define it (`vec.h` line 47),
  so its reference rounds the input to f16 and reads a table of f16
  results, as it does for GELU: this kernel computes the f32 formula and
  is compared to that table within test-backend-ops' NMSE, not bit for
  bit (as `gelu_f32` is).
-/

namespace Ggml.SlangCodegen.UnarySeeThrough

open LeanSlang
open Ggml.SlangCodegen.Common
open Ggml.SlangCodegen.Unary
open Ggml.SlangCodegen.ExactDivSqrt

/-- ggml-cpu's `GELU_QUICK_COEF`. -/
def geluQuickCoef : Float := -1.702

private def floatParam (n : String) : SlangBinding :=
  ⟨n, .scalar .float, Semantic.none, none, none, .qIn⟩

/-- `float gelu_quick(float x) { return x * (1 / (1 + exp(-1.702 x))); }`,
    ggml_gelu_quick_f32's association: the reciprocal first, then the
    product. -/
def fnGeluQuick : SlangFunctionDecl :=
  { retType := .scalar .float
  , name := "gelu_quick"
  , params := [floatParam "x"]
  , body := [ .ret (some (mul (v "x")
        (fdiv (fl 1.0) (add (fl 1.0) (.call "exp" [mul (fl geluQuickCoef) (v "x")]))))) ] }

def sqrtF32 : SlangShaderModule := shader [fnSqrtExact] (.call "sqrt_exact" [v "x"])
def geluQuickF32 : SlangShaderModule := shader [fnGeluQuick] (.call "gelu_quick" [v "x"])

/-- The kernels this module contributes, by their kernels.txt names. -/
def kernels : List (String × SlangShaderModule) :=
  [ ("sqrt_f32", sqrtF32)
  , ("gelu_quick_f32", geluQuickF32) ]

/-! ## Pins (siblings of silu_f32: `Unary.sibling`) -/

example : LeanSlang.emit sqrtF32 = sibling "float sqrt_exact(float x) {
  if ((!((x >= 5.421011e-20f) && (x <= 3.4028235e38f)))) {
    return sqrt(x);
  }
  precise float y0 = rsqrt(x);
  precise float g0 = (x * y0);
  precise float h0 = (0.5f * y0);
  precise float r0 = fma((-g0), g0, x);
  precise float g1 = fma(h0, r0, g0);
  precise float r1 = fma((-g1), g1, x);
  precise float g2 = fma(h0, r1, g1);
  uint bits = asuint(g2);
  precise float u_up = asfloat(((bits & 2139095040u) - 192937984u));
  precise float u_dn = (((bits & 8388607u) == 0u) ? (0.5f * u_up) : u_up);
  precise float e = fma((-g2), g2, x);
  precise float up = (e - (g2 * u_up));
  precise float dn = (e + (g2 * u_dn));
  precise float tu = ((0.25f * u_up) * u_up);
  precise float td = ((0.25f * u_dn) * u_dn);
  if ((up >= tu)) {
    return asfloat((bits + 1u));
  }
  if ((dn < td)) {
    return asfloat((bits - 1u));
  }
  return g2;
}
\n" "dst[d] = sqrt_exact(x);" := by native_decide

example : LeanSlang.emit geluQuickF32 = sibling "float gelu_quick(float x) {
  return (x * (1.0f / (1.0f + exp(((-1.702f) * x)))));
}

" "dst[d] = gelu_quick(x);" := by native_decide

example : sqrtF32.entryPointName = "main" ∧ geluQuickF32.entryPointName = "main" := by native_decide

end Ggml.SlangCodegen.UnarySeeThrough
