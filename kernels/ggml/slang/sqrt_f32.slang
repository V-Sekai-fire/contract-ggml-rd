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

float sqrt_exact(float x) {
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
  dst[d] = sqrt_exact(x);
}
