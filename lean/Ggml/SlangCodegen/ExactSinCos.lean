import LeanSlang
import Ggml.SlangCodegen.Common

/-!
# `Ggml.SlangCodegen.ExactSinCos` — the C runtime's `sinf` and `cosf`, bit for bit

XLA CPU's `sin` and `cos` are `sinf` and `cosf` of the C runtime (tests/xla_sincos): on the Windows desk
they are UCRT's, in `ucrtbase.dll` (`sinf` at RVA 0xAC360, `cosf` at 0xA7F70, the FMA path). They are
not a published algorithm, so this is a translation of the disassembly, and it equals the runtime for all
2^32 inputs (tests/xla_sincos/ucrt_libm.c, ucrt_libm_sweep.c).

Both compute in double precision with fused multiply-adds: a Taylor polynomial for small |x|; a reduction by
pi/2 with two constants for |x| below 0x4b800456 (`sinf`) or 0x4f490fdb (`cosf`); and above that a
Payne-Hanek reduction against a 176-byte table of 2/pi bits, with a 53 x 192-bit product. The cosine adds a
double-double product with pi/2 after that reduction. The table is read at byte offsets, so it is held as 22
64-bit words (a function cannot hold an array, as in ExactRsqrt) and joined by shifts.

Every intermediate is `precise`, so the compiler cannot contract or reorder what the runtime does not.
Needs SPIR-V Float64 and Int64.
-/

namespace Ggml.SlangCodegen.ExactSinCos

open LeanSlang
open Ggml.SlangCodegen.Common

private def f32 : SlangType := .scalar .float
private def f64 : SlangType := .scalar .double
private def u32 : SlangType := .scalar .uint
private def i32 : SlangType := .scalar .int
private def u64 : SlangType := .named "uint64_t"

private def dl (x : Float) : SlangExpr := .litDoubleExact x
private def c64 (n : Nat) : SlangExpr := .cast u64 (u n)
private def k64 (hi lo : Nat) : SlangExpr := .bin "|" (.bin "<<" (.cast u64 (u hi)) (u 32)) (.cast u64 (u lo))
private def bop (o : String) (a b : SlangExpr) : SlangExpr := .bin o a b
private def sub (a b : SlangExpr) : SlangExpr := .bin "-" a b
private def band (a b : SlangExpr) : SlangExpr := .bin "&" a b
private def bor (a b : SlangExpr) : SlangExpr := .bin "|" a b
private def bxor (a b : SlangExpr) : SlangExpr := .bin "^" a b
private def shl (a b : SlangExpr) : SlangExpr := .bin "<<" a b
private def shr (a b : SlangExpr) : SlangExpr := .bin ">>" a b
private def lt (a b : SlangExpr) : SlangExpr := .bin "<" a b
private def ge (a b : SlangExpr) : SlangExpr := .bin ">=" a b
private def eq (a b : SlangExpr) : SlangExpr := .bin "==" a b
private def ne (a b : SlangExpr) : SlangExpr := .bin "!=" a b
private def neg (a : SlangExpr) : SlangExpr := .un "-" a
private def fm (a b c : SlangExpr) : SlangExpr := .call "fma" [a, b, c]
private def pd (n : String) (e : SlangExpr) : SlangStmt := .declarePrecise f64 n (some e)
private def inP (n : String) (t : SlangType) : SlangBinding := ⟨n, t, Semantic.none, none, none, .qIn⟩
private def outP (n : String) (t : SlangType) : SlangBinding := ⟨n, t, Semantic.none, none, none, .qOut⟩
private def k63 : SlangExpr := shl (c64 1) (u 63)
private def k52 : SlangExpr := shl (c64 1) (u 52)

/-- The 176 bytes at ucrtbase.dll RVA 0x107360 as little-endian 64-bit words, (high, low) halves. -/
def phWords : Array (Nat × Nat) := #[(1948342284, 3239834080), (1269411821, 2126806581), (3320640540, 3890096458), (2830488734, 3944722327), (652033304, 4254759295), (3028137177, 4215054124), (1614789820, 1177380905), (2699853794, 1610577942), (3889328657, 2117007332), (4177847879, 1491471692), (303277681, 3231371645), (2633736712, 3621042925), (1215559799, 2893202598), (1266686244, 3148359494), (361159825, 3102279991), (1091041605, 2393864943), (1996713354, 1449455886), (1420292253, 1598543059), (166886, 230242576), (0, 0), (4293918720, 0), (0, 0)]

partial def wordTree (lo hi : Nat) : List SlangStmt :=
  if hi - lo ≤ 1 then
    let w := phWords.getD lo (0, 0)
    [.ret (some (k64 w.1 w.2))]
  else
    let mid := (lo + hi) / 2
    [.ifThen (lt (v "k") (u mid)) (wordTree lo mid) (wordTree mid hi)]

def fnD2B : SlangFunctionDecl :=
  { retType := u64, name := "ucrt_d2b", params := [inP "x" f64]
  , body := [ .declare u32 "lo" none, .declare u32 "hi" none
            , .expr (.call "asuint" [v "x", v "lo", v "hi"])
            , .ret (some (bor (shl (.cast u64 (v "hi")) (u 32)) (.cast u64 (v "lo")))) ] }

def fnB2D : SlangFunctionDecl :=
  { retType := f64, name := "ucrt_b2d", params := [inP "b" u64]
  , body := [ .ret (some (.call "asdouble" [.cast u32 (v "b"), .cast u32 (shr (v "b") (u 32))])) ] }

def fnMulhi : SlangFunctionDecl :=
  { retType := u64, name := "ucrt_mulhi", params := [inP "a" u64, inP "b" u64]
  , body :=
      [ .declare u64 "m" (some (c64 4294967295))
      , .declare u64 "a0" (some (band (v "a") (v "m"))), .declare u64 "a1" (some (shr (v "a") (u 32)))
      , .declare u64 "b0" (some (band (v "b") (v "m"))), .declare u64 "b1" (some (shr (v "b") (u 32)))
      , .declare u64 "p00" (some (mul (v "a0") (v "b0"))), .declare u64 "p01" (some (mul (v "a0") (v "b1")))
      , .declare u64 "p10" (some (mul (v "a1") (v "b0"))), .declare u64 "p11" (some (mul (v "a1") (v "b1")))
      , .declare u64 "mid" (some (add (add (shr (v "p00") (u 32)) (band (v "p01") (v "m"))) (band (v "p10") (v "m"))))
      , .ret (some (add (add (add (v "p11") (shr (v "p01") (u 32))) (shr (v "p10") (u 32))) (shr (v "mid") (u 32)))) ] }

/-- Index of the highest set bit of a 64-bit value. -/
def fnHibit : SlangFunctionDecl :=
  { retType := i32, name := "ucrt_hibit", params := [inP "x" u64]
  , body :=
      [ .declare u32 "h" (some (.cast u32 (shr (v "x") (u 32))))
      , .ifThen (ne (v "h") (u 0)) [ .ret (some (add (.litInt 32) (.cast i32 (.call "firstbithigh" [v "h"])))) ] []
      , .ret (some (.cast i32 (.call "firstbithigh" [.cast u32 (v "x")]))) ] }

def fnWord : SlangFunctionDecl :=
  { retType := u64, name := "ucrt_word", params := [inP "k" u32], body := wordTree 0 phWords.size }

/-- The 64-bit word at byte offset `o` of the table. -/
def fnTbl : SlangFunctionDecl :=
  { retType := u64, name := "ucrt_tbl", params := [inP "o" u32]
  , body :=
      [ .declare u32 "w" (some (shr (v "o") (u 3)))
      , .declare u32 "s" (some (mul (band (v "o") (u 7)) (u 8)))
      , .ifThen (eq (v "s") (u 0)) [ .ret (some (.call "ucrt_word" [v "w"])) ] []
      , .ret (some (bor (shr (.call "ucrt_word" [v "w"]) (v "s"))
                       (shl (.call "ucrt_word" [add (v "w") (u 1)]) (sub (u 64) (v "s"))))) ] }

/-- Payne-Hanek: |x| >= 2^24 as a double; n & 3, and the reduced argument in units of pi/2 as `hi` and `lo`. -/
def fnPh : SlangFunctionDecl :=
  { retType := .scalar .uint, name := "ucrt_ph"
  , params := [inP "ab" f64, outP "hi" f64, outP "lo" f64]
  , body :=
      [ .declare u64 "bits" (some (.call "ucrt_d2b" [v "ab"]))
      , .declare i32 "e" (some (sub (.cast i32 (shr (v "bits") (u 52))) (.litInt 1023)))
      , .declare i32 "off" (some (sub (.litInt 134) (shr (v "e") (u 3))))
      , .declare u64 "m" (some (bor (band (v "bits") (k64 0xfffff 0xffffffff)) k52))
      , .declare u64 "ta" (some (.call "ucrt_tbl" [.cast u32 (v "off")]))
      , .declare u64 "tb" (some (.call "ucrt_tbl" [add (.cast u32 (v "off")) (u 8)]))
      , .declare u64 "tc" (some (.call "ucrt_tbl" [add (.cast u32 (v "off")) (u 16)]))
      , .declare u64 "r8" (some (mul (v "ta") (v "m")))
      , .declare u64 "hiA" (some (.call "ucrt_mulhi" [v "ta", v "m"]))
      , .declare u64 "lowB" (some (mul (v "tb") (v "m")))
      , .declare u64 "r9" (some (add (v "lowB") (v "hiA")))
      , .declare u64 "r10" (some (add (.call "ucrt_mulhi" [v "tb", v "m"]) (.ternary (lt (v "r9") (v "lowB")) (c64 1) (c64 0))))
      , .assign (v "r10") (add (v "r10") (mul (v "tc") (v "m")))
      , .declare u32 "e7" (some (band (.cast u32 (v "e")) (u 7)))
      , .declare u32 "cnt" (some (sub (u 54) (v "e7")))
      , .declare u64 "rax" (some (shr (v "r10") (v "cnt")))
      , .declare u64 "cf" (some (band (shr (v "r10") (sub (v "cnt") (u 1))) (c64 1)))
      , .declare u64 "rdx" (some (c64 0))
      , .ifThen (ne (v "cf") (c64 0))
          [ .assign (v "r10") (.un "~" (v "r10")), .assign (v "r9") (.un "~" (v "r9"))
          , .assign (v "r8") (.un "~" (v "r8")), .assign (v "rdx") k63 ] []
      , .assign (v "rax") (add (v "rax") (v "cf"))
      , .declare u32 "n3" (some (band (.cast u32 (v "rax")) (u 3)))
      , .declare u32 "c2" (some (add (v "e7") (u 10)))
      , .assign (v "r10") (shr (shl (v "r10") (v "c2")) (v "c2"))
      , .declare i32 "r11" (some (sub (.cast i32 (v "c2")) (.litInt 64)))
      , .ifThen (eq (v "r10") (c64 0))
          [ .assign (v "r10") (v "r9"), .assign (v "r9") (v "r8"), .assign (v "r8") (c64 0)
          , .assign (v "r11") (sub (v "r11") (.litInt 64)) ] []
      , .declare i32 "hb" (some (.call "ucrt_hibit" [v "r10"]))
      , .assign (v "r11") (add (v "r11") (v "hb"))
      , .declare i32 "sh" (some (sub (v "hb") (.litInt 52)))
      , .ifThen (lt (v "sh") (.litInt 0))
          [ .declare u32 "s2" (some (.cast u32 (neg (v "sh"))))
          , .declare u64 "a" (some (v "r9"))
          , .assign (v "r10") (shl (v "r10") (v "s2")), .assign (v "r9") (shl (v "r9") (v "s2"))
          , .declare u32 "back" (some (sub (u 64) (v "s2")))
          , .assign (v "r10") (bor (v "r10") (shr (v "a") (v "back")))
          , .assign (v "r9") (bor (v "r9") (shr (v "r8") (v "back"))) ]
          [ .ifThen (bop ">" (v "sh") (.litInt 0))
              [ .declare u32 "s2" (some (.cast u32 (v "sh")))
              , .declare u64 "t8" (some (v "r10"))
              , .assign (v "r10") (shr (v "r10") (v "s2")), .assign (v "r9") (shr (v "r9") (v "s2"))
              , .assign (v "r9") (bor (v "r9") (shl (v "t8") (sub (u 64) (v "s2")))) ] [] ]
      , .assign (v "r11") (add (v "r11") (.litInt 1023))
      , .assign (v "r10") (bor (bor (band (v "r10") (.un "~" k52)) (v "rdx")) (shl (.cast u64 (v "r11")) (u 52)))
      , .assign (v "hi") (.call "ucrt_b2d" [v "r10"])
      , .assign (v "lo") (dl 0.0)
      , .ifThen (ne (v "r9") (c64 0))
          [ .declare u32 "lz1" (some (add (.cast u32 (sub (.litInt 63) (.call "ucrt_hibit" [v "r9"]))) (u 1)))
          , .declare u64 "mant" (some (.ternary (ge (v "lz1") (u 64)) (c64 0) (shr (shl (v "r9") (v "lz1")) (u 12))))
          , .declare i32 "ex" (some (sub (v "r11") (.cast i32 (add (v "lz1") (u 52)))))
          , .assign (v "lo") (.call "ucrt_b2d" [bor (bor (v "mant") (v "rdx")) (shl (.cast u64 (v "ex")) (u 52))]) ] []
      , .ret (some (v "n3")) ] }

def fnSinPoly : SlangFunctionDecl :=
  { retType := f64, name := "ucrt_sin_poly", params := [inP "r" f64]
  , body :=
      [ pd "x2" (mul (v "r") (v "r"))
      , pd "p" (fm (v "x2") (dl 2.7557319223985893e-06) (dl (-0.0001984126984126984)))
      , pd "p2" (fm (v "x2") (v "p") (dl 0.008333333333333333))
      , pd "p3" (fm (v "x2") (v "p2") (dl (-0.16666666666666666)))
      , pd "t" (mul (v "r") (v "x2"))
      , .ret (some (fm (v "p3") (v "t") (v "r"))) ] }

def fnCosPoly : SlangFunctionDecl :=
  { retType := f64, name := "ucrt_cos_poly", params := [inP "r" f64]
  , body :=
      [ pd "x2" (mul (v "r") (v "r"))
      , pd "h" (mul (v "x2") (dl 0.5))
      , pd "a" (sub (dl 1.0) (v "h"))
      , pd "b" (fm (v "x2") (dl (-2.755731922398589e-07)) (dl 2.4801587301587298e-05))
      , pd "b2" (fm (v "x2") (v "b") (dl (-0.0013888888888888887)))
      , pd "b3" (fm (v "x2") (v "b2") (dl 0.041666666666666664))
      , pd "x4" (mul (v "x2") (v "x2"))
      , .ret (some (fm (v "b3") (v "x4") (v "a"))) ] }

/-- sinf's tail: the polynomial for the quadrant, then the sign from the quadrant and from x. -/
def fnSinFinish : SlangFunctionDecl :=
  { retType := f64, name := "ucrt_sin_finish", params := [inP "r" f64, inP "n3" u32, inP "xs" f64]
  , body :=
      [ .declare f64 "res" none
      , .ifThen (eq (band (v "n3") (u 1)) (u 1))
          [ .assign (v "res") (.call "ucrt_cos_poly" [v "r"]) ] [ .assign (v "res") (.call "ucrt_sin_poly" [v "r"]) ]
      , .declare u64 "flip" (some (.ternary (.bin "||" (eq (v "n3") (u 2)) (eq (v "n3") (u 3))) (c64 0) k63))
      , .declare u64 "sx" (some (band (.un "~" (.call "ucrt_d2b" [v "xs"])) k63))
      , .ret (some (.call "ucrt_b2d" [bxor (bxor (.call "ucrt_d2b" [v "res"]) (v "flip")) (v "sx")])) ] }

/-- Shared by both: the quadrant and reduced argument for 0x3f490fdb < |x| < `limit`. -/
private def reduceNear (limit : Nat) : List SlangStmt :=
  [ pd "t" (fm (v "ab") (dl 0.6366197723675814) (dl 0.5))
  , .declare i32 "n" (some (.cast i32 (v "t")))
  , .assign (v "n3") (band (.cast u32 (v "n")) (u 3))
  , pd "nd" (.cast f64 (v "n"))
  , pd "r2" (fm (neg (v "nd")) (dl 1.5707963267341256) (v "ab"))
  , pd "nl" (mul (v "nd") (dl 6.077100506506192e-11))
  , .assign (v "r") (sub (v "r2") (v "nl")) ]

private def nanOrInf : List SlangStmt :=
  [ .ifThen (ge (v "ax") (u 0x7f800000))
      [ .ret (some (.ternary (bop ">" (v "ax") (u 0x7f800000))
          (.call "asfloat" [bor (v "xb") (u 0x400000)]) (.call "asfloat" [u 0xffc00000]))) ] [] ]

def fnSinf : SlangFunctionDecl :=
  { retType := f32, name := "ucrt_sinf", params := [inP "xf" f32]
  , body :=
      [ .declare u32 "xb" (some (.call "asuint" [v "xf"]))
      , .declare u32 "ax" (some (band (v "xb") (u 0x7fffffff))) ] ++ nanOrInf ++
      [ .declare f64 "xs" (some (.cast f64 (v "xf")))
      , .declare f64 "ab" (some (.call "abs" [v "xs"]))
      , .ifThen (bop ">" (v "ax") (u 0x3f490fdb))
          [ .declare u32 "n3" (some (u 0)), .declare f64 "r" (some (dl 0.0))
          , .ifThen (lt (v "ax") (u 0x4b800456))
              (reduceNear 0x4b800456 ++ [ .ret (some (.cast f32 (.call "ucrt_sin_finish" [v "r", v "n3", v "xs"]))) ])
              [ .declare f64 "hi" (some (dl 0.0)), .declare f64 "lo" (some (dl 0.0))
              , .assign (v "n3") (.call "ucrt_ph" [v "ab", v "hi", v "lo"])
              , pd "rr" (mul (v "hi") (dl 1.5707963267948966))
              , .ret (some (.cast f32 (.call "ucrt_sin_finish" [v "rr", v "n3", v "xs"]))) ] ] []
      , .ifThen (ge (v "ax") (u 0x3c000000)) [ .ret (some (.cast f32 (.call "ucrt_sin_poly" [v "xs"]))) ] []
      , .ifThen (ge (v "ax") (u 0x39000000))
          [ pd "x2" (mul (v "xs") (v "xs")), pd "t3" (mul (v "x2") (v "xs"))
          , .ret (some (.cast f32 (fm (neg (v "t3")) (dl 0.16666666666666666) (v "xs")))) ] []
      , .ret (some (v "xf")) ] }

def fnCosf : SlangFunctionDecl :=
  { retType := f32, name := "ucrt_cosf", params := [inP "xf" f32]
  , body :=
      [ .declare u32 "xb" (some (.call "asuint" [v "xf"]))
      , .declare u32 "ax" (some (band (v "xb") (u 0x7fffffff))) ] ++ nanOrInf ++
      [ .declare f64 "xs" (some (.cast f64 (v "xf")))
      , .declare f64 "ab" (some (.call "abs" [v "xs"]))
      , .ifThen (bop ">" (v "ax") (u 0x3f490fdb))
          ([ .declare u32 "n3" (some (u 0)), .declare f64 "r" (some (dl 0.0))
           , .ifThen (lt (v "ax") (u 0x4f490fdb))
              (reduceNear 0x4f490fdb)
              [ .declare f64 "hi" (some (dl 0.0)), .declare f64 "lo" (some (dl 0.0))
              , .assign (v "n3") (.call "ucrt_ph" [v "ab", v "hi", v "lo"])
              , .declare u64 "hb" (some (band (.call "ucrt_d2b" [v "hi"]) (k64 0xffffffff 0xf8000000)))
              , pd "htop" (.call "ucrt_b2d" [v "hb"])
              , pd "hlow" (sub (v "hi") (v "htop"))
              , pd "p" (mul (v "hi") (dl 1.5707963267948966))
              , pd "q0" (mul (v "htop") (dl 1.5707963109016418))
              , pd "q1" (sub (v "q0") (v "p"))
              , pd "q2" (fm (v "hlow") (dl 1.5707963109016418) (v "q1"))
              , pd "q3" (fm (v "htop") (dl 1.5893254712295857e-08) (v "q2"))
              , pd "q4" (fm (v "hlow") (dl 1.5893254712295857e-08) (v "q3"))
              , pd "w0" (mul (v "lo") (dl 1.5707963267948966))
              , pd "w1" (fm (v "hi") (dl 6.123233995736765e-17) (v "w0"))
              , pd "q5" (add (v "q4") (v "w1"))
              , .assign (v "r") (add (v "p") (v "q5")) ]
           , .declare f64 "res" none
           , .ifThen (eq (band (v "n3") (u 1)) (u 1))
              [ .assign (v "res") (.call "ucrt_sin_poly" [v "r"]) ] [ .assign (v "res") (.call "ucrt_cos_poly" [v "r"]) ]
           , .declare u64 "sg" (some (shl (shr (add (.cast u64 (v "n3")) (c64 1)) (u 1)) (u 63)))
           , .ret (some (.cast f32 (.call "ucrt_b2d" [bxor (.call "ucrt_d2b" [v "res"]) (v "sg")]))) ]) []
      , .ifThen (ge (v "ax") (u 0x3c000000)) [ .ret (some (.cast f32 (.call "ucrt_cos_poly" [v "xs"]))) ] []
      , .ifThen (ge (v "ax") (u 0x39000000))
          [ pd "hx" (mul (v "xs") (dl 0.5))
          , .ret (some (.cast f32 (fm (neg (v "xs")) (v "hx") (dl 1.0)))) ] []
      , .ret (some (.call "asfloat" [u 0x3f800000])) ] }

def sinHelpers : List SlangFunctionDecl :=
  [fnD2B, fnB2D, fnMulhi, fnHibit, fnWord, fnTbl, fnPh, fnSinPoly, fnCosPoly, fnSinFinish, fnSinf]

def cosHelpers : List SlangFunctionDecl :=
  [fnD2B, fnB2D, fnMulhi, fnHibit, fnWord, fnTbl, fnPh, fnSinPoly, fnCosPoly, fnCosf]

end Ggml.SlangCodegen.ExactSinCos
