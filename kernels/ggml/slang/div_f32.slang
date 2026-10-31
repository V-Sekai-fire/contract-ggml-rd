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
RWStructuredBuffer<float> s1;
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

float div_exact(float a, float b) {
  float aa = abs(a);
  float bb = abs(b);
  float qh = (a / b);
  float aq = abs(qh);
  if ((!((((aa >= 7.888609e-31f) && (aa <= 1.2676506e30f)) && ((bb >= 7.888609e-31f) && (bb <= 1.2676506e30f))) && ((aq >= 7.888609e-31f) && (aq <= 1.2676506e30f))))) {
    return qh;
  }
  precise float y0 = (1.0f / bb);
  precise float e = fma((-bb), y0, 1.0f);
  precise float y1 = fma(y0, e, y0);
  precise float q0 = (aa * y1);
  precise float r = fma((-bb), q0, aa);
  precise float q1 = fma(r, y1, q0);
  uint bits = asuint(q1);
  precise float u_up = asfloat(((bits & 2139095040u) - 192937984u));
  precise float u_dn = (((bits & 8388607u) == 0u) ? (0.5f * u_up) : u_up);
  precise float r2 = fma((-q1), bb, aa);
  precise float tu = ((0.5f * u_up) * bb);
  precise float td = ((0.5f * u_dn) * bb);
  uint out = bits;
  if ((r2 >= tu)) {
    out = (bits + 1u);
  } else {
    if ((r2 < (-td))) {
      out = (bits - 1u);
    }
  }
  return asfloat((out | ((asuint(a) ^ asuint(b)) & 2147483648u)));
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
  uint b = off4(19u, (i0 % pw(19u)), (i1 % pw(20u)), (i2 % pw(21u)), (i3 % pw(22u)));
  dst[d] = div_exact(s0[a], s1[b]);
}
