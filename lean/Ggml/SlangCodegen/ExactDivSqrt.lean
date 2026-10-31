import LeanSlang
import Ggml.SlangCodegen.Common

/-!
# `Ggml.SlangCodegen.ExactDivSqrt` — correctly rounded f32 divide and square root

Vulkan lets a divide be 2.5 ULP and a square root 2 ULP off the IEEE value, and the RTX 3090 returns
a result one ulp away in about a quarter of the elements. ggml-cpu computes `a / b` and `sqrtf(x)`
as IEEE operations, so the helpers here finish a hardware estimate with fused multiply-adds and
settle it by an exact comparison against the midpoint to each neighbour:

- `div_exact(a, b)`: an estimate `y0 = 1/|b|` is refined to `y1`, `q = |a| y1` is corrected by the
  exact remainder `fma(-b, q, a)`, and the result moves one ulp when `a / b` is past the midpoint
  (the remainder against `ulp(q) |b| / 2`). The sign is `sign(a) xor sign(b)`.
- `sqrt_exact(x)`: an estimate from `rsqrt(x)` is refined twice, then moved one ulp when `x` is past
  the square of the midpoint (the residual `fma(-g, g, x)` against `g ulp + ulp^2 / 4`).

Outside the range where every intermediate is a normal number, and for zero, subnormals, negatives,
infinities and NaNs, they return the hardware result, which is what the bare operators gave.
`sqrt_exact` holds for x in [2^-64, FLT_MAX] and `div_exact` for |a|, |b| and |a / b| in
[2^-100, 2^100]. Every statement is `precise` (SPIR-V NoContraction), so no optimiser may fuse or
reassociate a step: the algorithm is only correct with each product and sum rounded as written.

The same code was checked on the host with `fmaf` before it was written here: `sqrt_exact` for all
2^32 bit patterns and `div_exact` over 6 * 10^9 random pairs and every special value, with the
hardware estimate displaced by up to 3 ulp each way (`tests/exact_div_sqrt`).
-/

namespace Ggml.SlangCodegen.ExactDivSqrt

open LeanSlang
open Ggml.SlangCodegen.Common

private def f32 : SlangType := .scalar .float
private def u32 : SlangType := .scalar .uint

private def fparam (n : String) : SlangBinding :=
  ⟨n, f32, Semantic.none, none, none, .qIn⟩

/-- `precise float name = init;` -/
private def pf (name : String) (init : SlangExpr) : SlangStmt := .declarePrecise f32 name (some init)
private def fl (x : Float) : SlangExpr := .litFloatExact x
private def fdiv (a b : SlangExpr) : SlangExpr := .bin "/" a b
private def fma (a b c : SlangExpr) : SlangExpr := .call "fma" [a, b, c]
private def neg (e : SlangExpr) : SlangExpr := .un "-" e
private def not (e : SlangExpr) : SlangExpr := .un "!" e
private def both (a b : SlangExpr) : SlangExpr := .bin "&&" a b
private def ge (a b : SlangExpr) : SlangExpr := .bin ">=" a b
private def le (a b : SlangExpr) : SlangExpr := .bin "<=" a b
private def lt (a b : SlangExpr) : SlangExpr := .bin "<" a b
private def eq (a b : SlangExpr) : SlangExpr := .bin "==" a b
private def sub (a b : SlangExpr) : SlangExpr := .bin "-" a b
private def band (a b : SlangExpr) : SlangExpr := .bin "&" a b
private def bor (a b : SlangExpr) : SlangExpr := .bin "|" a b
private def bxor (a b : SlangExpr) : SlangExpr := .bin "^" a b
private def asuint (e : SlangExpr) : SlangExpr := .call "asuint" [e]
private def asfloat (e : SlangExpr) : SlangExpr := .call "asfloat" [e]
private def absE (e : SlangExpr) : SlangExpr := .call "abs" [e]
private def inRange (x lo hi : SlangExpr) : SlangExpr := both (ge x lo) (le x hi)

/-- The smallest `x` for which `sqrt_exact` runs: 2^-64. -/
def sqrtLo : Float := 5.421010862427522e-20
/-- The largest finite f32. -/
def fltMax : Float := 3.4028234663852886e38
/-- The range of `|a|`, `|b|` and `|a / b|` for which `div_exact` runs: 2^-100 to 2^100. -/
def divLo : Float := 7.888609052210118e-31
def divHi : Float := 1.2676506002282294e30

/-- 23 << 23: subtracting it from an exponent field gives the ulp of a float. -/
def ulpShift : Nat := 0x0B800000

/-- `ulp(g)` upward, and downward (half of it when `g` is a power of two), from the bits of `g`. -/
private def ulps : List SlangStmt :=
  [ pf "u_up" (asfloat (sub (band (v "bits") (u 0x7f800000)) (u ulpShift)))
  , pf "u_dn" (.ternary (eq (band (v "bits") (u 0x007fffff)) (u 0)) (mul (fl 0.5) (v "u_up")) (v "u_up")) ]

def fnSqrtExact : SlangFunctionDecl :=
  { retType := f32
  , name := "sqrt_exact"
  , params := [fparam "x"]
  , body :=
      [ .ifThen (not (inRange (v "x") (fl sqrtLo) (fl fltMax)))
          [ .ret (some (.call "sqrt" [v "x"])) ] []
      , pf "y0" (.call "rsqrt" [v "x"])
      , pf "g0" (mul (v "x") (v "y0"))
      , pf "h0" (mul (fl 0.5) (v "y0"))
      , pf "r0" (fma (neg (v "g0")) (v "g0") (v "x"))
      , pf "g1" (fma (v "h0") (v "r0") (v "g0"))
      , pf "r1" (fma (neg (v "g1")) (v "g1") (v "x"))
      , pf "g2" (fma (v "h0") (v "r1") (v "g1"))
      , .declare u32 "bits" (some (asuint (v "g2"))) ]
    ++ ulps ++
      [ pf "e" (fma (neg (v "g2")) (v "g2") (v "x"))
      , pf "up" (sub (v "e") (mul (v "g2") (v "u_up")))
      , pf "dn" (add (v "e") (mul (v "g2") (v "u_dn")))
      , pf "tu" (mul (mul (fl 0.25) (v "u_up")) (v "u_up"))
      , pf "td" (mul (mul (fl 0.25) (v "u_dn")) (v "u_dn"))
      , .ifThen (ge (v "up") (v "tu")) [ .ret (some (asfloat (add (v "bits") (u 1)))) ] []
      , .ifThen (lt (v "dn") (v "td")) [ .ret (some (asfloat (sub (v "bits") (u 1)))) ] []
      , .ret (some (v "g2")) ] }

def fnDivExact : SlangFunctionDecl :=
  { retType := f32
  , name := "div_exact"
  , params := [fparam "a", fparam "b"]
  , body :=
      [ .declare f32 "aa" (some (absE (v "a")))
      , .declare f32 "bb" (some (absE (v "b")))
      , .declare f32 "qh" (some (fdiv (v "a") (v "b")))
      , .declare f32 "aq" (some (absE (v "qh")))
      , .ifThen (not (both (both (inRange (v "aa") (fl divLo) (fl divHi)) (inRange (v "bb") (fl divLo) (fl divHi)))
                           (inRange (v "aq") (fl divLo) (fl divHi))))
          [ .ret (some (v "qh")) ] []
      , pf "y0" (fdiv (fl 1.0) (v "bb"))
      , pf "e" (fma (neg (v "bb")) (v "y0") (fl 1.0))
      , pf "y1" (fma (v "y0") (v "e") (v "y0"))
      , pf "q0" (mul (v "aa") (v "y1"))
      , pf "r" (fma (neg (v "bb")) (v "q0") (v "aa"))
      , pf "q1" (fma (v "r") (v "y1") (v "q0"))
      , .declare u32 "bits" (some (asuint (v "q1"))) ]
    ++ ulps ++
      [ pf "r2" (fma (neg (v "q1")) (v "bb") (v "aa"))
      , pf "tu" (mul (mul (fl 0.5) (v "u_up")) (v "bb"))
      , pf "td" (mul (mul (fl 0.5) (v "u_dn")) (v "bb"))
      , .declare u32 "out" (some (v "bits"))
      , .ifThen (ge (v "r2") (v "tu"))
          [ .assign (v "out") (add (v "bits") (u 1)) ]
          [ .ifThen (lt (v "r2") (neg (v "td"))) [ .assign (v "out") (sub (v "bits") (u 1)) ] [] ]
      , .ret (some (asfloat (bor (v "out") (band (bxor (asuint (v "a")) (asuint (v "b"))) (u 0x80000000))))) ] }

end Ggml.SlangCodegen.ExactDivSqrt
