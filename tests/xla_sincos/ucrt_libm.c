// UCRT's sinf and cosf on the FMA path (ucrtbase.dll, sinf at RVA 0xAC360, cosf at 0xA7F70), translated from the disassembly.
// Double precision: Taylor polynomials, a two-constant reduction by pi/2, Payne-Hanek for large arguments.
#include <math.h>
#include <stdint.h>
#include <string.h>

static const uint8_t ph_table[176] = {
	0xe0, 0xf1, 0x1b, 0xc1, 0x0c, 0x58, 0x21, 0x74, 0x35, 0x7e, 0xc4, 0x7e, 0xed, 0xaf, 0xa9, 0x4b,
	0x4a, 0x29, 0xde, 0xe7, 0x1c, 0xf4, 0xec, 0xc5, 0x97, 0xaf, 0x1f, 0xeb, 0x9e, 0xd4, 0xb5, 0xa8,
	0x7f, 0x79, 0x9a, 0xfd, 0x18, 0x3d, 0xdd, 0x26, 0x2c, 0x9f, 0x3c, 0xfb, 0xd9, 0xb4, 0x7d, 0xb4,
	0x29, 0x68, 0x2d, 0x46, 0xbc, 0xbc, 0x3f, 0x60, 0x16, 0x78, 0xff, 0x5f, 0xe2, 0x7f, 0xec, 0xa0,
	0xe4, 0xf7, 0x2e, 0x7e, 0x11, 0x72, 0xd2, 0xe7, 0x4c, 0x0d, 0xe6, 0x58, 0x47, 0xe6, 0x04, 0xf9,
	0x7d, 0xd1, 0x9a, 0xc0, 0x71, 0xa6, 0x13, 0x12, 0xed, 0xba, 0xd4, 0xd7, 0x08, 0xa2, 0xfb, 0x9c,
	0xa6, 0xc4, 0x72, 0xac, 0x77, 0xf8, 0x73, 0x48, 0x46, 0x27, 0xa8, 0xbb, 0x24, 0x19, 0x80, 0x4b,
	0x37, 0x09, 0xe9, 0xb8, 0x91, 0xdc, 0x86, 0x15, 0xef, 0x7a, 0xaf, 0x8e, 0x45, 0xf9, 0x07, 0x41,
	0x0e, 0xf1, 0x64, 0x56, 0x8a, 0x6d, 0x03, 0x77, 0xd3, 0xd4, 0x47, 0x5f, 0x9d, 0xf0, 0xa7, 0x54,
	0x10, 0x39, 0xb9, 0x0d, 0xe6, 0x8b, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};
static uint64_t ld64(int off) { uint64_t v; memcpy(&v, ph_table + off, 8); return v; }
static double bits2d(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }
static uint64_t d2bits(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }
static uint32_t f2bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

#define K_1_6 0.16666666666666666
#define K_1_7F (-0.0001984126984126984)
#define K_1_9F 2.7557319223985893e-06
#define K_1_5F 0.008333333333333333
#define K_M1_6 (-0.16666666666666666)
#define K_2PI 0.6366197723675814
#define K_PIO2_HI 1.5707963267341256
#define K_PIO2_LO 6.077100506506192e-11
#define K_PIO2 1.5707963267948966
#define K_M05 (-0.5)
#define K_1_8F 2.4801587301587298e-05
#define K_M1_10F (-2.755731922398589e-07)
#define K_M1_720 (-0.0013888888888888887)
#define K_1_24 0.041666666666666664

// The result r (a double in about [-pi/4, pi/4]) and n&3, as the instructions leave them.
static double finish(double r, uint64_t n3, double xs) {
	double res;
	double x2 = r * r;
	if ((n3 & 1) == 1) {
		double a = fma(x2, K_M05, 1.0);
		double b = fma(x2, K_M1_10F, K_1_8F);
		b = fma(x2, b, K_M1_720);
		b = fma(x2, b, K_1_24);
		double x4 = x2 * x2;
		res = fma(b, x4, a);
	} else {
		double p = fma(x2, K_1_9F, K_1_7F);
		p = fma(x2, p, K_1_5F);
		p = fma(x2, p, K_M1_6);
		double t = r * x2;
		res = fma(p, t, r);
	}
	uint64_t flip = (n3 == 2 || n3 == 3) ? 0 : 0x8000000000000000ull; // vandnpd
	uint64_t sx = (~d2bits(xs)) & 0x8000000000000000ull;
	return bits2d(d2bits(res) ^ flip ^ sx);
}

// Payne-Hanek: |x| as a double in, n&3 and the reduced argument as hi (an f64 with the leading 53 bits) and lo (the next ones).
static void ph_reduce(double ab, uint64_t *n3p, double *hi, double *lo) {
	uint64_t bitsd = d2bits(ab);
	int64_t e = (int64_t)(bitsd >> 52) - 0x3ff;
	int64_t off = 0x86 - (e >> 3);
	uint64_t m = (bitsd & 0xfffffffffffffull) | (1ull << 52);
	unsigned __int128 pa = (unsigned __int128)ld64((int)off) * m;
	uint64_t r8 = (uint64_t)pa, hiA = (uint64_t)(pa >> 64);
	uint64_t B = ld64((int)off + 8), C = ld64((int)off + 16);
	unsigned __int128 pb = (unsigned __int128)B * m;
	uint64_t e7 = (uint64_t)e & 7;
	unsigned __int128 s = (unsigned __int128)(uint64_t)pb + hiA;
	uint64_t r9 = (uint64_t)s;
	uint64_t r10 = (uint64_t)(pb >> 64) + (uint64_t)(s >> 64);
	unsigned __int128 pc = (unsigned __int128)C * m;
	r10 += (uint64_t)pc;
	unsigned cnt = (unsigned)(0x36 - e7);
	uint64_t rax = r10 >> cnt;
	unsigned cf = (unsigned)((r10 >> (cnt - 1)) & 1);
	uint64_t rdx = 0;
	if (cf) { r10 = ~r10; r9 = ~r9; r8 = ~r8; rdx = 0x8000000000000000ull; }
	rax += cf;
	*n3p = rax & 3;
	unsigned c2 = (unsigned)(e7 + 10);
	r10 = (r10 << c2) >> c2;
	int64_t r11 = (int64_t)(c2) - 0x40;
	int64_t hb;
	if (r10 == 0) { r10 = r9; r9 = r8; r8 = 0; r11 -= 0x40; }
	hb = 63 - __builtin_clzll(r10);
	r11 += hb;
	int64_t sh = hb - 0x34;
	if (sh < 0) {
		unsigned s2 = (unsigned)(-sh);
		uint64_t a = r9;
		r10 <<= s2; r9 <<= s2;
		unsigned back = 64 - s2;
		r10 |= a >> back;
		r9 |= r8 >> back;
	} else if (sh > 0) {
		unsigned s2 = (unsigned)sh;
		uint64_t t8 = r10;
		r10 >>= s2; r9 >>= s2;
		r9 |= t8 << (64 - s2);
	}
	r11 += 0x3ff;
	r10 &= ~(1ull << 52);
	r10 |= rdx;
	r10 |= (uint64_t)r11 << 52;
	*hi = bits2d(r10 | 0);
	{
		uint64_t l9 = r9;
		uint64_t lbits = 0;
		if (l9 != 0) {
			unsigned lz1 = (unsigned)(__builtin_clzll(l9) + 1);
			uint64_t mant = lz1 >= 64 ? 0 : ((l9 << lz1) >> 12);
			int64_t ex = (int64_t)(r11 - 0) - (int64_t)(lz1 + 52);
			lbits = mant | rdx | ((uint64_t)ex << 52);
		}
		*lo = bits2d(lbits);
	}
}

float ucrt_sinf(float xf) {
	uint32_t ax = f2bits(xf) & 0x7fffffffu;
	if (ax >= 0x7f800000u) return xf - xf;
	double xs = (double)xf;
	double ab = bits2d(d2bits(xs) & 0x7fffffffffffffffull);
	if (ax > 0x3f490fdbu) {
		if (ax < 0x4b800456u) {
			double t = fma(ab, K_2PI, 0.5);
			int32_t n = (int32_t)t;
			uint64_t n3 = (uint64_t)(int64_t)n & 3;
			double nd = (double)n;
			double r2 = fma(-nd, K_PIO2_HI, ab);
			double r = r2 - nd * K_PIO2_LO;
			return (float)finish(r, n3, xs);
		}
		uint64_t n3; double hi, lo;
		ph_reduce(ab, &n3, &hi, &lo);
		return (float)finish(hi * K_PIO2, n3, xs);
	}
	if (ax >= 0x3c000000u) {
		double x2 = xs * xs;
		double p = fma(x2, K_1_9F, K_1_7F);
		p = fma(x2, p, K_1_5F);
		p = fma(x2, p, K_M1_6);
		double t = xs * x2;
		return (float)fma(p, t, xs);
	}
	if (ax >= 0x39000000u) {
		double x2 = xs * xs;
		double t = x2 * xs;
		return (float)fma(-t, K_1_6, xs);
	}
	return xf;
}

// cosf: the same polynomials; the sign is ((n + 1) >> 1) & 1, and the reduced argument is as the instructions leave it.
static double cos_poly(double r) {
	double x2 = r * r;
	double a = 1.0 - x2 * 0.5;
	double b = fma(x2, K_M1_10F, K_1_8F);
	b = fma(x2, b, K_M1_720);
	b = fma(x2, b, K_1_24);
	double x4 = x2 * x2;
	return fma(b, x4, a);
}
static double sin_poly(double r) {
	double x2 = r * r;
	double p = fma(x2, K_1_9F, K_1_7F);
	p = fma(x2, p, K_1_5F);
	p = fma(x2, p, K_M1_6);
	double t = r * x2;
	return fma(p, t, r);
}

float ucrt_cosf(float xf) {
	uint32_t ax = f2bits(xf) & 0x7fffffffu;
	if (ax >= 0x7f800000u) return xf - xf;
	double xs = (double)xf;
	double ab = bits2d(d2bits(xs) & 0x7fffffffffffffffull);
	if (ax > 0x3f490fdbu) {
		uint64_t n3; double r;
		if (ax < 0x4f490fdbu) {
			double t = fma(ab, K_2PI, 0.5);
			int32_t n = (int32_t)t;
			n3 = (uint64_t)(int64_t)n & 3;
			double nd = (double)n;
			double r2 = fma(-nd, K_PIO2_HI, ab);
			r = r2 - nd * K_PIO2_LO;
		} else {
			double hi, lo;
			ph_reduce(ab, &n3, &hi, &lo);
			double hi_top = bits2d(d2bits(hi) & 0xfffffffff8000000ull);
			double hi_low = hi - hi_top;
			double p = hi * K_PIO2;
			double q = hi_top * 1.5707963109016418;
			q = q - p;
			q = fma(hi_low, 1.5707963109016418, q);
			q = fma(hi_top, 1.5893254712295857e-08, q);
			q = fma(hi_low, 1.5893254712295857e-08, q);
			double w = lo * K_PIO2;
			w = fma(hi, 6.123233995736765e-17, w);
			q = q + w;
			r = p + q;
		}
		double res = (n3 & 1) ? sin_poly(r) : cos_poly(r);
		uint64_t sgn = (((n3 + 1) >> 1) << 63);
		return (float)bits2d(d2bits(res) ^ sgn);
	}
	if (ax >= 0x3c000000u) return (float)cos_poly(xs);
	if (ax >= 0x39000000u) return (float)fma(-xs, xs * 0.5, 1.0);
	return 1.0f;
}
