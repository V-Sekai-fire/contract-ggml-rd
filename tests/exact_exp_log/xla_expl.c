// Host model of XLA CPU's f32 exp and log (xla/backends/cpu/codegen/polynomial_approximations.cc at d867969800d4).
// Built with -ffp-contract=off so every fusion is explicit. FUSE is a bit set picking which mul+add pairs
// become one fmaf, to match what LLVM did with the contract flag.
#include <math.h>
#include <stdint.h>
#include <string.h>

static float bits2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t f2bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

#define F_MULADD 1   /* MulAdd(a,b,c) = fmaf(a,b,c) */
#define F_SUBMUL 2   /* x - c*n as fmaf(-c, n, x) */
#define F_LASTADD 4  /* log: the q1*e, half*x2, q2*e adds fused */

static inline float ftz(float f) {
	uint32_t u = f2bits(f);
	return (u & 0x7f800000u) == 0 ? bits2f(u & 0x80000000u) : f;
}

static inline float ma(int fuse, float a, float b, float c) {
	return (fuse & F_MULADD) ? fmaf(a, b, c) : c + a * b;
}

float xla_exp1(float input, int fuse) {
	input = ftz(input);
	const float ln2_hi = 0.693145751953125f, ln2_lo = 1.4286067653301877e-6f;
	const float p0 = 1.9875691500E-4f, p1 = 1.3981999507E-3f, p2 = 8.3334519073E-3f, p3 = 4.1665795894E-2f, p4 = 1.6666665459E-1f, p5 = 5.0000001201E-1f;
	float x = input;
	/* Clamp: select(uge(a,low), a, low) then select(ule(a,high), a, high); NaN stays NaN */
	x = (x >= -87.8f || x != x) ? x : -87.8f;
	x = (x <= 88.8f || x != x) ? x : 88.8f;
	float n = floorf(ma(fuse, x, 1.44269504088896341f, 0.5f));
	n = (n >= -127.0f || n != n) ? n : -127.0f;
	n = (n <= 128.0f || n != n) ? n : 128.0f;
	if (fuse & F_SUBMUL) {
		x = fmaf(-ln2_hi, n, x);
		x = fmaf(-ln2_lo, n, x);
	} else {
		x = x - ln2_hi * n;
		x = x - ln2_lo * n;
	}
	float z = ma(fuse, x, p0, p1);
	z = ma(fuse, z, x, p2);
	z = ma(fuse, z, x, p3);
	z = ma(fuse, z, x, p4);
	z = ma(fuse, z, x, p5);
	z = ma(fuse, z, x * x, x);
	z = 1.0f + z;
	int32_t ni = (n != n) ? INT32_MIN : (int32_t)n; /* cvttss2si */
	uint32_t shl = (uint32_t)(ni + 0x7f) << 23;
	uint32_t pow2 = shl < 0x7f000000u ? shl : 0x7f000000u;
	uint32_t rem = shl - pow2;
	float zs = bits2f(f2bits(z) + rem);
	return ftz(zs * bits2f(pow2));
}

float xla_log1(float input, int fuse) {
	input = ftz(input);
	const float SQRTHF = 0.707106781186547524f;
	const float p0 = 7.0376836292E-2f, p1 = -1.1514610310E-1f, p2 = 1.1676998740E-1f, p3 = -1.2420140846E-1f, p4 = 1.4249322787E-1f;
	const float p5 = -1.6668057665E-1f, p6 = 2.0000714765E-1f, p7 = -2.4999993993E-1f, p8 = 3.3333331174E-1f;
	const float q1 = -2.12194440e-4f, q2 = 0.693359375f;
	int invalid = !(input > 0.0f);          /* ule(input, 0): true when unordered too */
	int is_zero = input == 0.0f;
	int is_pinf = input == INFINITY;
	float tmp0; /* Max(min_norm_pos, input) fast: input > min ? input : min */
	{
		float mn = bits2f(0x00800000u);
		tmp0 = (mn > input) ? mn : input; /* x86 maxss(min, input) semantics: returns input when unordered */
		if (input != input) tmp0 = input;
	}
	int32_t emm0 = (int32_t)(f2bits(tmp0) >> 23);
	tmp0 = bits2f((f2bits(tmp0) & ~0x7f800000u) | f2bits(0.5f));
	emm0 -= 0x7f;
	float e = 1.0f + (float)emm0;
	int mask = tmp0 < SQRTHF;
	float tmp1 = mask ? tmp0 : 0.0f;
	tmp0 = tmp0 - 1.0f;
	e = e - (mask ? 1.0f : 0.0f);
	tmp0 = tmp0 + tmp1;
	float x2 = tmp0 * tmp0, x3 = x2 * tmp0;
	float y, y1, y2;
	y = ma(fuse, tmp0, p0, p1);
	y1 = ma(fuse, tmp0, p3, p4);
	y2 = ma(fuse, tmp0, p6, p7);
	y = ma(fuse, y, tmp0, p2);
	y1 = ma(fuse, y1, tmp0, p5);
	y2 = ma(fuse, y2, tmp0, p8);
	y = ma(fuse, y, x3, y1);
	y = ma(fuse, y, x3, y2);
	/* bits 8,16,32 pick the fusion of the tail: 8 = y*x3 + q1*e as fma(q1,e,y*x3); 16 = fma(y,x3,q1*e); 32 = q2*e fused */
	{
		float yp = y;
		float t2 = 0.5f * x2;
		float yy;
		if (fuse & 8) yy = fmaf(q1, e, yp * x3);
		else if (fuse & 16) yy = fmaf(yp, x3, q1 * e);
		else yy = yp * x3 + q1 * e;
		tmp0 = tmp0 - t2;
		tmp0 = tmp0 + yy;
		tmp0 = (fuse & 32) ? fmaf(q2, e, tmp0) : tmp0 + q2 * e;
	}
	if (is_zero) return -INFINITY;
	if (is_pinf) return INFINITY;
	if (invalid) return bits2f(0xffffffffu); /* all-ones OR: a NaN */
	return tmp0;
}

__declspec(dllexport) void exp_many(const uint32_t *in, uint32_t *out, long n, int fuse) {
	for (long i = 0; i < n; ++i) out[i] = f2bits(xla_exp1(bits2f(in[i]), fuse));
}
__declspec(dllexport) void log_many(const uint32_t *in, uint32_t *out, long n, int fuse) {
	for (long i = 0; i < n; ++i) out[i] = f2bits(xla_log1(bits2f(in[i]), fuse));
}
