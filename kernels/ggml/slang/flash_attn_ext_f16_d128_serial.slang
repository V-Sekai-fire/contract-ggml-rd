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

float pf(uint k) {
  return asfloat(pw(k));
}

uint off4(uint w, uint i0, uint i1, uint i2, uint i3) {
  return ((((pw((w + 8u)) + (i0 * pw((w + 4u)))) + (i1 * pw((w + 5u)))) + (i2 * pw((w + 6u)))) + (i3 * pw((w + 7u))));
}

float ld_f16_s1(uint e) {
  uint w = s1[(e >> 1u)];
  uint h = (((e & 1u) != 0u) ? (w >> 16u) : (w & 65535u));
  uint s = ((h & 32768u) << 16u);
  uint x = (h & 31744u);
  uint b = (((h & 32767u) << 13u) + 939524096u);
  if ((x == 31744u)) {
    b = (b + 939524096u);
  }
  if ((x == 0u)) {
    b = asuint((asfloat(b) - asfloat(947912704u)));
  }
  return asfloat((b | s));
}

float ld_f16_s2(uint e) {
  uint w = s2[(e >> 1u)];
  uint h = (((e & 1u) != 0u) ? (w >> 16u) : (w & 65535u));
  uint s = ((h & 32768u) << 16u);
  uint x = (h & 31744u);
  uint b = (((h & 32767u) << 13u) + 939524096u);
  if ((x == 31744u)) {
    b = (b + 939524096u);
  }
  if ((x == 0u)) {
    b = asuint((asfloat(b) - asfloat(947912704u)));
  }
  return asfloat((b | s));
}

uint f32_to_f16(float f) {
  uint u = asuint(f);
  uint s = ((u >> 16u) & 32768u);
  uint a = (u & 2147483647u);
  if ((a > 2139095040u)) {
    return (s | 32256u);
  }
  if ((a >= 1199566848u)) {
    return (s | 31744u);
  }
  if ((a >= 947912704u)) {
    uint h = ((a - 939524096u) >> 13u);
    uint r = (a & 8191u);
    if (((r > 4096u) || ((r == 4096u) && ((h & 1u) != 0u)))) {
      h = (h + 1u);
    }
    return (s | h);
  }
  if ((a < 855638017u)) {
    return s;
  }
  uint m = ((a & 8388607u) | 8388608u);
  uint sh = (126u - (a >> 23u));
  uint h = (m >> sh);
  uint r = (m & ((1u << sh) - 1u));
  uint hw = (1u << (sh - 1u));
  if (((r > hw) || ((r == hw) && ((h & 1u) != 0u)))) {
    h = (h + 1u);
  }
  return (s | h);
}

float f16_bits_to_f32(uint h) {
  uint s = ((h & 32768u) << 16u);
  uint x = (h & 31744u);
  uint b = (((h & 32767u) << 13u) + 939524096u);
  if ((x == 31744u)) {
    b = (b + 939524096u);
  }
  if ((x == 0u)) {
    b = asuint((asfloat(b) - asfloat(947912704u)));
  }
  return asfloat((b | s));
}

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint e = ((((gid.y * pw(54u)) + gid.x) * 64u) + lid.x);
  if ((e >= pw(53u))) {
    return;
  }
  uint nq = pw(11u);
  uint iq = (e % nq);
  uint rr = (e / nq);
  uint h = (rr % pw(12u));
  uint b = (rr / pw(12u));
  uint lk = pw(20u);
  uint hk = (h / pw(55u));
  uint bk = (b / pw(56u));
  uint hv = (h / pw(57u));
  uint bv = (b / pw(58u));
  float scale = pf(59u);
  float cap = pf(60u);
  uint qb = off4(10u, 0u, iq, h, b);
  uint kb = off4(19u, 0u, 0u, hk, bk);
  uint kn = pw(24u);
  uint vb = off4(28u, 0u, 0u, hv, bv);
  uint vn = pw(33u);
  float qv[128];
  float acc[128];
  float sc[8];
  for (uint c = 0u; c < 128u; ++c) {
    qv[c] = f16_bits_to_f32(f32_to_f16(s0[(qb + c)]));
    acc[c] = 0.000000;
  }
  float m = asfloat((4286578688u | (pw(0u) & 0u)));
  float ls = 0.000000;
  uint nt = ((lk + 7u) / 8u);
  for (uint tt = 0u; tt < nt; ++tt) {
    uint k0 = (tt * 8u);
    for (uint j = 0u; j < 8u; ++j) {
      uint key = (k0 + j);
      float s = asfloat((4286578688u | (pw(0u) & 0u)));
      if ((key < lk)) {
        s = 0.000000;
        for (uint c = 0u; c < 128u; ++c) {
          s = (s + (qv[c] * ld_f16_s1(((kb + (key * kn)) + c))));
        }
        s = (s * scale);
        if ((cap != 0.000000)) {
          s = (cap * tanh(s));
        }
      }
      sc[j] = s;
    }
    float mt = m;
    for (uint j = 0u; j < 8u; ++j) {
      mt = max(mt, sc[j]);
    }
    float corr = exp((m - mt));
    for (uint c = 0u; c < 128u; ++c) {
      acc[c] = (acc[c] * corr);
    }
    float ps = 0.000000;
    for (uint j = 0u; j < 8u; ++j) {
      float p = exp((sc[j] - mt));
      ps = (ps + p);
      uint key = (k0 + j);
      if ((key < lk)) {
        for (uint c = 0u; c < 128u; ++c) {
          acc[c] = (acc[c] + (p * ld_f16_s2(((vb + (key * vn)) + c))));
        }
      }
    }
    ls = ((ls * corr) + ps);
    m = mt;
  }
  float inv = ((ls == 0.000000) ? 0.000000 : (1.000000 / ls));
  for (uint c = 0u; c < 128u; ++c) {
    dst[off4(1u, c, h, iq, b)] = (acc[c] * inv);
  }
}
