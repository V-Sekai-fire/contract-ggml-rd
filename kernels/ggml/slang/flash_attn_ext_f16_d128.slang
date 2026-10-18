struct Slot {
  uint base;
  uint pad0;
  uint pad1;
  uint pad2;
};

groupshared float Qs[8][129];
groupshared float Ks[8][129];
groupshared float Vs[8][128];
groupshared float Ss[8][9];

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
  uint t = lid.x;
  uint r = (t / 8u);
  uint l = (t % 8u);
  uint iq = ((gid.x * 8u) + r);
  uint h = gid.y;
  uint b = gid.z;
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
  bool valid = (iq < pw(11u));
  for (uint i = 0u; i < 16u; ++i) {
    uint c = (l + (i * 8u));
    float x = 0.000000;
    if (valid) {
      x = f16_bits_to_f32(f32_to_f16(s0[(qb + c)]));
    }
    Qs[r][c] = x;
  }
  float m = asfloat((4286578688u | (pw(0u) & 0u)));
  float ls = 0.000000;
  float acc0 = 0.000000;
  float acc1 = 0.000000;
  float acc2 = 0.000000;
  float acc3 = 0.000000;
  float acc4 = 0.000000;
  float acc5 = 0.000000;
  float acc6 = 0.000000;
  float acc7 = 0.000000;
  float acc8 = 0.000000;
  float acc9 = 0.000000;
  float acc10 = 0.000000;
  float acc11 = 0.000000;
  float acc12 = 0.000000;
  float acc13 = 0.000000;
  float acc14 = 0.000000;
  float acc15 = 0.000000;
  uint nt = ((lk + 7u) / 8u);
  for (uint tt = 0u; tt < nt; ++tt) {
    uint k0 = (tt * 8u);
    GroupMemoryBarrierWithGroupSync();
    for (uint i = 0u; i < 16u; ++i) {
      uint f = (t + (i * 64u));
      uint j = (f / 128u);
      uint c = (f % 128u);
      uint key = (k0 + j);
      float kx = 0.000000;
      float vx = 0.000000;
      if ((key < lk)) {
        kx = ld_f16_s1(((kb + (key * kn)) + c));
        vx = ld_f16_s2(((vb + (key * vn)) + c));
      }
      Ks[j][c] = kx;
      Vs[j][c] = vx;
    }
    GroupMemoryBarrierWithGroupSync();
    float d0 = 0.000000;
    for (uint c = 0u; c < 128u; ++c) {
      float qc = Qs[r][c];
      d0 = (d0 + (qc * Ks[l][c]));
    }
    d0 = (d0 * scale);
    if ((cap != 0.000000)) {
      d0 = (cap * tanh(d0));
    }
    if (((k0 + l) >= lk)) {
      d0 = asfloat((4286578688u | (pw(0u) & 0u)));
    }
    Ss[r][l] = d0;
    GroupMemoryBarrierWithGroupSync();
    float mt = m;
    for (uint j = 0u; j < 8u; ++j) {
      mt = max(mt, Ss[r][j]);
    }
    float corr = exp((m - mt));
    acc0 = (acc0 * corr);
    acc1 = (acc1 * corr);
    acc2 = (acc2 * corr);
    acc3 = (acc3 * corr);
    acc4 = (acc4 * corr);
    acc5 = (acc5 * corr);
    acc6 = (acc6 * corr);
    acc7 = (acc7 * corr);
    acc8 = (acc8 * corr);
    acc9 = (acc9 * corr);
    acc10 = (acc10 * corr);
    acc11 = (acc11 * corr);
    acc12 = (acc12 * corr);
    acc13 = (acc13 * corr);
    acc14 = (acc14 * corr);
    acc15 = (acc15 * corr);
    float ps = 0.000000;
    for (uint j = 0u; j < 8u; ++j) {
      float p = exp((Ss[r][j] - mt));
      ps = (ps + p);
      acc0 = (acc0 + (p * Vs[j][l]));
      acc1 = (acc1 + (p * Vs[j][(l + 8u)]));
      acc2 = (acc2 + (p * Vs[j][(l + 16u)]));
      acc3 = (acc3 + (p * Vs[j][(l + 24u)]));
      acc4 = (acc4 + (p * Vs[j][(l + 32u)]));
      acc5 = (acc5 + (p * Vs[j][(l + 40u)]));
      acc6 = (acc6 + (p * Vs[j][(l + 48u)]));
      acc7 = (acc7 + (p * Vs[j][(l + 56u)]));
      acc8 = (acc8 + (p * Vs[j][(l + 64u)]));
      acc9 = (acc9 + (p * Vs[j][(l + 72u)]));
      acc10 = (acc10 + (p * Vs[j][(l + 80u)]));
      acc11 = (acc11 + (p * Vs[j][(l + 88u)]));
      acc12 = (acc12 + (p * Vs[j][(l + 96u)]));
      acc13 = (acc13 + (p * Vs[j][(l + 104u)]));
      acc14 = (acc14 + (p * Vs[j][(l + 112u)]));
      acc15 = (acc15 + (p * Vs[j][(l + 120u)]));
    }
    ls = ((ls * corr) + ps);
    m = mt;
  }
  if (valid) {
    float inv = ((ls == 0.000000) ? 0.000000 : (1.000000 / ls));
    dst[off4(1u, l, h, iq, b)] = (acc0 * inv);
    dst[off4(1u, (l + 8u), h, iq, b)] = (acc1 * inv);
    dst[off4(1u, (l + 16u), h, iq, b)] = (acc2 * inv);
    dst[off4(1u, (l + 24u), h, iq, b)] = (acc3 * inv);
    dst[off4(1u, (l + 32u), h, iq, b)] = (acc4 * inv);
    dst[off4(1u, (l + 40u), h, iq, b)] = (acc5 * inv);
    dst[off4(1u, (l + 48u), h, iq, b)] = (acc6 * inv);
    dst[off4(1u, (l + 56u), h, iq, b)] = (acc7 * inv);
    dst[off4(1u, (l + 64u), h, iq, b)] = (acc8 * inv);
    dst[off4(1u, (l + 72u), h, iq, b)] = (acc9 * inv);
    dst[off4(1u, (l + 80u), h, iq, b)] = (acc10 * inv);
    dst[off4(1u, (l + 88u), h, iq, b)] = (acc11 * inv);
    dst[off4(1u, (l + 96u), h, iq, b)] = (acc12 * inv);
    dst[off4(1u, (l + 104u), h, iq, b)] = (acc13 * inv);
    dst[off4(1u, (l + 112u), h, iq, b)] = (acc14 * inv);
    dst[off4(1u, (l + 120u), h, iq, b)] = (acc15 * inv);
  }
}
