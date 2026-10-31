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

float exp_xla(float x) {
  if (isnan(x)) {
    return x;
  }
  uint xb = asuint(x);
  precise float xi = (((xb & 2139095040u) == 0u) ? asfloat((xb & 2147483648u)) : x);
  precise float c0 = ((xi < (-87.8f)) ? (-87.8f) : xi);
  precise float c1 = ((c0 > 88.8f) ? 88.8f : c0);
  precise float n0 = floor(fma(c1, 1.442695f, 0.5f));
  precise float n1 = ((n0 < (-127.0f)) ? (-127.0f) : n0);
  precise float n = ((n1 > 128.0f) ? 128.0f : n1);
  precise float a0 = fma((-0.69314575f), n, c1);
  precise float a = fma((-1.4286068e-6f), n, a0);
  precise float z0 = fma(a, 0.00019875691f, 0.0013981999f);
  precise float z1 = fma(z0, a, 0.008333452f);
  precise float z2 = fma(z1, a, 0.041665796f);
  precise float z3 = fma(z2, a, 0.16666666f);
  precise float z4 = fma(z3, a, 0.5f);
  precise float aa = (a * a);
  precise float z5 = fma(z4, aa, a);
  precise float z = (1.0f + z5);
  uint shl = (uint((int(n) + 127)) << 23u);
  uint p2 = min(shl, 2130706432u);
  uint remb = (shl - p2);
  precise float zs = asfloat((asuint(z) + remb));
  precise float r = (zs * asfloat(p2));
  return (((asuint(r) & 2139095040u) == 0u) ? asfloat((asuint(r) & 2147483648u)) : r);
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
  dst[d] = exp_xla(x);
}
