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

float log_xla(float x) {
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
  dst[d] = log_xla(x);
}
