/* The exact functions the shaders will contain, with the GPU's initial approximation modelled as the
   correctly rounded value displaced by k ulps. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float f_of(uint32_t u) { float x; memcpy(&x, &u, 4); return x; }
static uint32_t u_of(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }
static float hw_k(float v, int k) { return f_of((uint32_t)((int64_t)u_of(v) + k)); }

static uint64_t subnormal_uses = 0;
static float watch(float v) {
	if (v != 0.0f && fabsf(v) < 1.17549435e-38f) ++subnormal_uses;
	return v;
}

#define SQRT_LO 0x1.0p-64f
#define SQRT_HI 3.40282347e38f
#define DIV_LO 0x1.0p-100f
#define DIV_HI 0x1.0p100f

static float sqrt_exact(float x, int k) {
	if (!(x >= SQRT_LO && x <= SQRT_HI)) return sqrtf(x);
	float y0 = hw_k((float)(1.0 / sqrt((double)x)), k);
	float g0 = x * y0;
	float h0 = 0.5f * y0;
	float r0 = fmaf(-g0, g0, x);
	float g1 = fmaf(h0, r0, g0);
	float r1 = fmaf(-g1, g1, x);
	float g2 = fmaf(h0, r1, g1);
	uint32_t bits = u_of(g2);
	float u_up = f_of((bits & 0x7f800000u) - 0x0B800000u);
	float u_dn = (bits & 0x007fffffu) == 0u ? 0.5f * u_up : u_up;
	float e = fmaf(-g2, g2, x);
	float up = e - g2 * u_up;
	float dn = e + g2 * u_dn;
	float tu = watch(0.25f * u_up * u_up);
	float td = watch(0.25f * u_dn * u_dn);
	if (up >= tu) return f_of(bits + 1u);
	if (dn < td) return f_of(bits - 1u);
	return g2;
}

static float div_exact(float a, float b, int k) {
	float aa = fabsf(a), bb = fabsf(b);
	float qh = a / b;
	float aq = fabsf(qh);
	if (!(aa >= DIV_LO && aa <= DIV_HI && bb >= DIV_LO && bb <= DIV_HI && aq >= DIV_LO && aq <= DIV_HI)) return qh;
	float y0 = hw_k((float)(1.0 / (double)bb), k);
	float e = fmaf(-bb, y0, 1.0f);
	float y1 = fmaf(y0, e, y0);
	float q0 = aa * y1;
	float r = fmaf(-bb, q0, aa);
	float q1 = fmaf(r, y1, q0);
	uint32_t bits = u_of(q1);
	float u_up = f_of((bits & 0x7f800000u) - 0x0B800000u);
	float u_dn = (bits & 0x007fffffu) == 0u ? 0.5f * u_up : u_up;
	float r2 = fmaf(-q1, bb, aa);
	float tu = watch(0.5f * u_up * bb);
	float td = watch(0.5f * u_dn * bb);
	uint32_t out = bits;
	if (r2 >= tu) out = bits + 1u;
	else if (r2 < -td) out = bits - 1u;
	return f_of(out | ((u_of(a) ^ u_of(b)) & 0x80000000u));
}

static uint64_t rng_state = 88172645463325252ull;
static uint32_t rnd(void) {
	rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
	return (uint32_t)(rng_state >> 16);
}

int main(int argc, char **argv) {
	int k = argc > 1 ? atoi(argv[1]) : 0;
	uint64_t bad = 0, total = 0;
	/* sqrt: every float, including zero, subnormals, negatives, infinities and NaNs */
	for (uint64_t u = 0; u <= 0xffffffffull; ++u) {
		float x = f_of((uint32_t)u);
		float want = sqrtf(x), got = sqrt_exact(x, k);
		++total;
		int same = u_of(got) == u_of(want) || (isnan(got) && isnan(want));
		if (!same && ++bad <= 5) printf("  sqrt x=%a want=%a got=%a\n", x, want, got);
	}
	printf("sqrt k=%d every float: total=%llu wrong=%llu subnormal_threshold_uses=%llu\n", k, (unsigned long long)total, (unsigned long long)bad, (unsigned long long)subnormal_uses);
	uint64_t sq_bad = bad;
	/* divide: random sign, exponent over a wider range than the domain, random mantissa, and specials */
	bad = total = 0;
	for (uint64_t i = 0; i < 6000000000ull; ++i) {
		uint32_t ra = rnd(), rb = rnd();
		uint32_t ea = 127u - 110u + (ra >> 20) % 221u, eb = 127u - 110u + (rb >> 20) % 221u;
		float a = f_of((ra << 31 >> 0 & 0x80000000u) | (ea << 23) | (rnd() & 0x7fffffu));
		float b = f_of((rb << 31 >> 0 & 0x80000000u) | (eb << 23) | (rnd() & 0x7fffffu));
		float want = a / b, got = div_exact(a, b, k);
		++total;
		int same = u_of(got) == u_of(want) || (isnan(got) && isnan(want));
		if (!same && ++bad <= 5) printf("  div a=%a b=%a want=%a got=%a\n", a, b, want, got);
	}
	float sp[] = { 0.0f, -0.0f, INFINITY, -INFINITY, NAN, 1.0f, -1.0f, 1.17549435e-38f, 1e-45f, 3.4028235e38f };
	for (int i = 0; i < 10; ++i) for (int j = 0; j < 10; ++j) {
		float want = sp[i] / sp[j], got = div_exact(sp[i], sp[j], k);
		++total;
		int same = u_of(got) == u_of(want) || (isnan(got) && isnan(want));
		if (!same && ++bad <= 5) printf("  div special a=%a b=%a want=%a got=%a\n", sp[i], sp[j], want, got);
	}
	printf("div k=%d random across exponents 2^-110..2^110 plus specials: total=%llu wrong=%llu subnormal_threshold_uses=%llu\n", k, (unsigned long long)total, (unsigned long long)bad, (unsigned long long)subnormal_uses);
	return (bad | sq_bad) != 0;
}
