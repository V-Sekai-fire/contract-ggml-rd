struct Slot {
  uint base;
  uint pad0;
  uint pad1;
  uint pad2;
};

[[vk::binding(0, 0)]]
StructuredBuffer<uint> params;
[[vk::binding(1, 0)]]
RWStructuredBuffer<float> s0;
[[vk::binding(2, 0)]]
RWStructuredBuffer<uint> s1;
[[vk::binding(3, 0)]]
RWStructuredBuffer<uint> s2;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> dst;
[[vk::binding(0, 1)]]
ConstantBuffer<Slot> slot;

uint pw(uint k) {
  return params[(slot.base + k)];
}

void unravel4(uint e, uint w, out uint i0, out uint i1, out uint i2, out uint i3) {
  uint n0 = pw(w);
  uint n1 = pw((w + 1u));
  uint n2 = pw((w + 2u));
  i0 = (e % n0);
  uint r = (e / n0);
  i1 = (r % n1);
  r = (r / n1);
  i2 = (r % n2);
  i3 = (r / n2);
}

uint off4(uint w, uint i0, uint i1, uint i2, uint i3) {
  return ((((pw((w + 8u)) + (i0 * pw((w + 4u)))) + (i1 * pw((w + 5u)))) + (i2 * pw((w + 6u)))) + (i3 * pw((w + 7u))));
}

uint64_t ucrt_d2b(double x) {
  uint lo;
  uint hi;
  asuint(x, lo, hi);
  return ((uint64_t(hi) << 32u) | uint64_t(lo));
}

double ucrt_b2d(uint64_t b) {
  return asdouble(uint(b), uint((b >> 32u)));
}

uint64_t ucrt_mulhi(uint64_t a, uint64_t b) {
  uint64_t m = uint64_t(4294967295u);
  uint64_t a0 = (a & m);
  uint64_t a1 = (a >> 32u);
  uint64_t b0 = (b & m);
  uint64_t b1 = (b >> 32u);
  uint64_t p00 = (a0 * b0);
  uint64_t p01 = (a0 * b1);
  uint64_t p10 = (a1 * b0);
  uint64_t p11 = (a1 * b1);
  uint64_t mid = (((p00 >> 32u) + (p01 & m)) + (p10 & m));
  return (((p11 + (p01 >> 32u)) + (p10 >> 32u)) + (mid >> 32u));
}

int ucrt_hibit(uint64_t x) {
  uint h = uint((x >> 32u));
  if ((h != 0u)) {
    return (32 + int(firstbithigh(h)));
  }
  return int(firstbithigh(uint(x)));
}

uint64_t ucrt_word(uint k) {
  if ((k < 11u)) {
    if ((k < 5u)) {
      if ((k < 2u)) {
        if ((k < 1u)) {
          return ((uint64_t(1948342284u) << 32u) | uint64_t(3239834080u));
        } else {
          return ((uint64_t(1269411821u) << 32u) | uint64_t(2126806581u));
        }
      } else {
        if ((k < 3u)) {
          return ((uint64_t(3320640540u) << 32u) | uint64_t(3890096458u));
        } else {
          if ((k < 4u)) {
            return ((uint64_t(2830488734u) << 32u) | uint64_t(3944722327u));
          } else {
            return ((uint64_t(652033304u) << 32u) | uint64_t(4254759295u));
          }
        }
      }
    } else {
      if ((k < 8u)) {
        if ((k < 6u)) {
          return ((uint64_t(3028137177u) << 32u) | uint64_t(4215054124u));
        } else {
          if ((k < 7u)) {
            return ((uint64_t(1614789820u) << 32u) | uint64_t(1177380905u));
          } else {
            return ((uint64_t(2699853794u) << 32u) | uint64_t(1610577942u));
          }
        }
      } else {
        if ((k < 9u)) {
          return ((uint64_t(3889328657u) << 32u) | uint64_t(2117007332u));
        } else {
          if ((k < 10u)) {
            return ((uint64_t(4177847879u) << 32u) | uint64_t(1491471692u));
          } else {
            return ((uint64_t(303277681u) << 32u) | uint64_t(3231371645u));
          }
        }
      }
    }
  } else {
    if ((k < 16u)) {
      if ((k < 13u)) {
        if ((k < 12u)) {
          return ((uint64_t(2633736712u) << 32u) | uint64_t(3621042925u));
        } else {
          return ((uint64_t(1215559799u) << 32u) | uint64_t(2893202598u));
        }
      } else {
        if ((k < 14u)) {
          return ((uint64_t(1266686244u) << 32u) | uint64_t(3148359494u));
        } else {
          if ((k < 15u)) {
            return ((uint64_t(361159825u) << 32u) | uint64_t(3102279991u));
          } else {
            return ((uint64_t(1091041605u) << 32u) | uint64_t(2393864943u));
          }
        }
      }
    } else {
      if ((k < 19u)) {
        if ((k < 17u)) {
          return ((uint64_t(1996713354u) << 32u) | uint64_t(1449455886u));
        } else {
          if ((k < 18u)) {
            return ((uint64_t(1420292253u) << 32u) | uint64_t(1598543059u));
          } else {
            return ((uint64_t(166886u) << 32u) | uint64_t(230242576u));
          }
        }
      } else {
        if ((k < 20u)) {
          return ((uint64_t(0u) << 32u) | uint64_t(0u));
        } else {
          if ((k < 21u)) {
            return ((uint64_t(4293918720u) << 32u) | uint64_t(0u));
          } else {
            return ((uint64_t(0u) << 32u) | uint64_t(0u));
          }
        }
      }
    }
  }
}

uint64_t ucrt_tbl(uint o) {
  uint w = (o >> 3u);
  uint s = ((o & 7u) * 8u);
  if ((s == 0u)) {
    return ucrt_word(w);
  }
  return ((ucrt_word(w) >> s) | (ucrt_word((w + 1u)) << (64u - s)));
}

uint ucrt_ph(double ab, out double hi, out double lo) {
  uint64_t bits = ucrt_d2b(ab);
  int e = (int((bits >> 52u)) - 1023);
  int off = (134 - (e >> 3u));
  uint64_t m = ((bits & ((uint64_t(1048575u) << 32u) | uint64_t(4294967295u))) | (uint64_t(1u) << 52u));
  uint64_t ta = ucrt_tbl(uint(off));
  uint64_t tb = ucrt_tbl((uint(off) + 8u));
  uint64_t tc = ucrt_tbl((uint(off) + 16u));
  uint64_t r8 = (ta * m);
  uint64_t hiA = ucrt_mulhi(ta, m);
  uint64_t lowB = (tb * m);
  uint64_t r9 = (lowB + hiA);
  uint64_t r10 = (ucrt_mulhi(tb, m) + ((r9 < lowB) ? uint64_t(1u) : uint64_t(0u)));
  r10 = (r10 + (tc * m));
  uint e7 = (uint(e) & 7u);
  uint cnt = (54u - e7);
  uint64_t rax = (r10 >> cnt);
  uint64_t cf = ((r10 >> (cnt - 1u)) & uint64_t(1u));
  uint64_t rdx = uint64_t(0u);
  if ((cf != uint64_t(0u))) {
    r10 = (~r10);
    r9 = (~r9);
    r8 = (~r8);
    rdx = (uint64_t(1u) << 63u);
  }
  rax = (rax + cf);
  uint n3 = (uint(rax) & 3u);
  uint c2 = (e7 + 10u);
  r10 = ((r10 << c2) >> c2);
  int r11 = (int(c2) - 64);
  if ((r10 == uint64_t(0u))) {
    r10 = r9;
    r9 = r8;
    r8 = uint64_t(0u);
    r11 = (r11 - 64);
  }
  int hb = ucrt_hibit(r10);
  r11 = (r11 + hb);
  int sh = (hb - 52);
  if ((sh < 0)) {
    uint s2 = uint((-sh));
    uint64_t a = r9;
    r10 = (r10 << s2);
    r9 = (r9 << s2);
    uint back = (64u - s2);
    r10 = (r10 | (a >> back));
    r9 = (r9 | (r8 >> back));
  } else {
    if ((sh > 0)) {
      uint s2 = uint(sh);
      uint64_t t8 = r10;
      r10 = (r10 >> s2);
      r9 = (r9 >> s2);
      r9 = (r9 | (t8 << (64u - s2)));
    }
  }
  r11 = (r11 + 1023);
  r10 = (((r10 & (~(uint64_t(1u) << 52u))) | rdx) | (uint64_t(r11) << 52u));
  hi = ucrt_b2d(r10);
  lo = 0.0L;
  if ((r9 != uint64_t(0u))) {
    uint lz1 = (uint((63 - ucrt_hibit(r9))) + 1u);
    uint64_t mant = ((lz1 >= 64u) ? uint64_t(0u) : ((r9 << lz1) >> 12u));
    int ex = (r11 - int((lz1 + 52u)));
    lo = ucrt_b2d(((mant | rdx) | (uint64_t(ex) << 52u)));
  }
  return n3;
}

double ucrt_sin_poly(double r) {
  precise double x2 = (r * r);
  precise double p = fma(x2, 2.7557319223985893e-6L, asdouble(0x1A01A01Au, 0xBF2A01A0u));
  precise double p2 = fma(x2, p, asdouble(0x11111111u, 0x3F811111u));
  precise double p3 = fma(x2, p2, (-0.16666666666666666L));
  precise double t = (r * x2);
  return fma(p3, t, r);
}

double ucrt_cos_poly(double r) {
  precise double x2 = (r * r);
  precise double h = (x2 * 0.5L);
  precise double a = (1.0L - h);
  precise double b = fma(x2, (-2.755731922398589e-7L), asdouble(0x1A01A019u, 0x3EFA01A0u));
  precise double b2 = fma(x2, b, asdouble(0x16C16C16u, 0xBF56C16Cu));
  precise double b3 = fma(x2, b2, asdouble(0x55555555u, 0x3FA55555u));
  precise double x4 = (x2 * x2);
  return fma(b3, x4, a);
}

double ucrt_sin_finish(double r, uint n3, double xs) {
  double res;
  if (((n3 & 1u) == 1u)) {
    res = ucrt_cos_poly(r);
  } else {
    res = ucrt_sin_poly(r);
  }
  uint64_t flip = (((n3 == 2u) || (n3 == 3u)) ? uint64_t(0u) : (uint64_t(1u) << 63u));
  uint64_t sx = ((~ucrt_d2b(xs)) & (uint64_t(1u) << 63u));
  return ucrt_b2d(((ucrt_d2b(res) ^ flip) ^ sx));
}

float ucrt_sinf(float xf) {
  uint xb = asuint(xf);
  uint ax = (xb & 2147483647u);
  if ((ax >= 2139095040u)) {
    return ((ax > 2139095040u) ? asfloat((xb | 4194304u)) : asfloat(4290772992u));
  }
  double xs = double(xf);
  double ab = abs(xs);
  if ((ax > 1061752795u)) {
    uint n3 = 0u;
    double r = 0.0L;
    if ((ax < 1266680918u)) {
      precise double t = fma(ab, 0.6366197723675814L, 0.5L);
      int n = int(t);
      n3 = (uint(n) & 3u);
      precise double nd = double(n);
      precise double r2 = fma((-nd), 1.5707963267341256L, ab);
      precise double nl = (nd * 6.077100506506192e-11L);
      r = (r2 - nl);
      return float(ucrt_sin_finish(r, n3, xs));
    } else {
      double hi = 0.0L;
      double lo = 0.0L;
      n3 = ucrt_ph(ab, hi, lo);
      precise double rr = (hi * 1.5707963267948966L);
      return float(ucrt_sin_finish(rr, n3, xs));
    }
  }
  if ((ax >= 1006632960u)) {
    return float(ucrt_sin_poly(xs));
  }
  if ((ax >= 956301312u)) {
    precise double x2 = (xs * xs);
    precise double t3 = (x2 * xs);
    return float(fma((-t3), 0.16666666666666666L, xs));
  }
  return xf;
}

[shader("compute")] [numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint e = ((((gid.y * pw(54u)) + gid.x) * 256u) + lid.x);
  if ((e >= pw(53u))) {
    return;
  }
  uint i0;
  uint i1;
  uint i2;
  uint i3;
  unravel4(e, 1u, i0, i1, i2, i3);
  uint d = off4(1u, i0, i1, i2, i3);
  uint a = off4(10u, i0, i1, i2, i3);
  float x = s0[a];
  dst[d] = ucrt_sinf(x);
}
