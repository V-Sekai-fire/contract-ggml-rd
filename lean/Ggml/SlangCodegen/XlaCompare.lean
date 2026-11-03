import LeanSlang
import Ggml.SlangCodegen.Common
import Ggml.SlangCodegen.Binary

/-!
# `Ggml.SlangCodegen.XlaCompare` — XLA CPU's f32 `maximum`, `minimum` and `compare`, bit for bit

The StableHLO lowering (`guest/pjrt-rd/stablehlo.cpp`) needs these for values computed at run time (ReLU,
clamp, `jnp.where` of a run-time condition). Measured on XLA CPU (jax 0.12.0.1, an AMD Zen 2 and an Apple
M-series CPU, `guest/pjrt-rd/tools/minmax_xla.py`, 196 pairs of special values):

- the inputs are read with denormals as zero of the same sign (XLA runs with DAZ and FTZ set);
- `maximum` and `minimum` return a NaN if either input is one, and order -0 below +0;
- `compare` is the IEEE comparison of the inputs after that flush. `NE` is true for a NaN.

Every case of the model equals XLA (NaN payloads aside: x86 XLA ORs the two inputs' bits for `minimum`).
The flush is done on the exponent bits, so no driver's denormal mode matters. A compare writes 1.0 or 0.0,
which the lowering uses as an i1 value. The kernels have Binary's shape (one thread per element, ggml's
broadcast, every operand strided); `guest/ggml-rd/ops/xla_compare.cpp` packs them for `GGML_OP_MAP_CUSTOM2`.
-/

namespace Ggml.SlangCodegen.XlaCompare

open LeanSlang
open Ggml.SlangCodegen.Common
open Ggml.SlangCodegen.Binary

private def f32 : SlangType := .scalar .float
private def u32 : SlangType := .scalar .uint
private def fl (x : Float) : SlangExpr := .litFloatExact x
private def band (a b : SlangExpr) : SlangExpr := .bin "&" a b
private def asuint (e : SlangExpr) : SlangExpr := .call "asuint" [e]
private def asfloat (e : SlangExpr) : SlangExpr := .call "asfloat" [e]
private def p2 (a b : String) : List SlangBinding :=
  [⟨a, f32, Semantic.none, none, none, .qIn⟩, ⟨b, f32, Semantic.none, none, none, .qIn⟩]

/-- A denormal read as the zero of its sign, by the exponent bits. -/
def fnDaz : SlangFunctionDecl :=
  { retType := f32
  , name := "xla_daz"
  , params := [⟨"x", f32, Semantic.none, none, none, .qIn⟩]
  , body :=
      [ .declare u32 "b" (some (asuint (v "x")))
      , .ret (some (.ternary (.bin "==" (band (v "b") (u 0x7f800000)) (u 0))
          (asfloat (band (v "b") (u 0x80000000))) (v "x"))) ] }

/-- `maximum` (`isMin = false`) or `minimum`: NaN first, then equal values (the zeros) by sign. -/
def fnMinMax (isMin : Bool) : SlangFunctionDecl :=
  { retType := f32
  , name := if isMin then "xla_min" else "xla_max"
  , params := p2 "a0" "b0"
  , body :=
      [ .declare f32 "a" (some (.call "xla_daz" [v "a0"]))
      , .declare f32 "b" (some (.call "xla_daz" [v "b0"]))
      , .ifThen (.call "isnan" [v "a"]) [ .ret (some (v "a")) ] []
      , .ifThen (.call "isnan" [v "b"]) [ .ret (some (v "b")) ] []
      , .ifThen (.bin "==" (v "a") (v "b"))
          [ .ret (some (.ternary (.bin "!=" (.bin ">>" (asuint (v "a")) (u 31)) (u 0))
              (if isMin then v "a" else v "b") (if isMin then v "b" else v "a"))) ] []
      , .ret (some (.ternary (.bin (if isMin then "<" else ">") (v "a") (v "b")) (v "a") (v "b"))) ] }

/-- The six directions, by their StableHLO names, and the Slang operator of each. -/
def directions : List (String × String) :=
  [("gt", ">"), ("ge", ">="), ("lt", "<"), ("le", "<="), ("eq", "=="), ("ne", "!=")]

def cmpExpr (op : String) (x y : SlangExpr) : SlangExpr :=
  .ternary (.bin op (.call "xla_daz" [x]) (.call "xla_daz" [y])) (fl 1.0) (fl 0.0)

def minMaxShader (isMin : Bool) : SlangShaderModule :=
  kernelModule .float .float .uint .float
    [fnPw, fnUnravel4, fnOff4, fnDaz, fnMinMax isMin]
    (entry1D threadgroup (bodyWith (fun x y => .call (if isMin then "xla_min" else "xla_max") [x, y])))

def cmpShader (op : String) : SlangShaderModule :=
  kernelModule .float .float .uint .float
    [fnPw, fnUnravel4, fnOff4, fnDaz]
    (entry1D threadgroup (bodyWith (cmpExpr op)))

/-- The kernels this module contributes, by their kernels.txt names. -/
def kernels : List (String × SlangShaderModule) :=
  [ ("max_xla_f32", minMaxShader false)
  , ("min_xla_f32", minMaxShader true) ]
  ++ directions.map (fun (n, op) => ("cmp_" ++ n ++ "_xla_f32", cmpShader op))

/-- Pins: each kernel is Binary's add_f32 with the one store changed and the helpers added. -/
example : ((LeanSlang.emit (minMaxShader false)).splitOn "dst[d] = xla_max(s0[a], s1[b]);").length = 2 := by native_decide
example : ((LeanSlang.emit (minMaxShader true)).splitOn "dst[d] = xla_min(s0[a], s1[b]);").length = 2 := by native_decide
example : ((LeanSlang.emit (cmpShader ">")).splitOn "(xla_daz(s0[a]) > xla_daz(s1[b]))").length = 2 := by native_decide
example : kernels.length = 8 := by native_decide

end Ggml.SlangCodegen.XlaCompare
