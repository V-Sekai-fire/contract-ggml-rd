// Table-driven emulation of RSQRTPS for AMD Zen 2, in the style of Bochs's approximate_rsqrt (cpu/sse_rcp.cc).
// The tables hold the 12-bit mantissa fraction of the result for each of the top 12 mantissa bits of the input,
// one table per exponent parity. They are generated from the instruction itself by gen_zen2_table.py.
#include <stdint.h>
#include <string.h>
#include "../../guest/ggml-rd/rsqrt_zen2_table.h"

static float bits2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t f2bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

/* RSQRTPS on Zen 2 */
uint32_t zen2_approximate_rsqrt_bits(uint32_t op) {
	const uint32_t sign = op >> 31;
	const uint32_t exp = (op >> 23) & 0xff;
	const uint32_t fraction = op & 0x7fffff;
	if (exp == 0) {
		return (sign << 31) | 0x7f800000u; /* zero or denormal: signed infinity */
	}
	if (exp == 0xff) {
		if (fraction != 0) {
			return op | 0x00400000u; /* NaN: quieted */
		}
		return sign ? 0xffc00000u : 0u; /* -inf: default NaN; +inf: zero */
	}
	if (sign) {
		return 0xffc00000u; /* negative: default NaN */
	}
	/* x = 1.f * 2^(exp-127); 1/sqrt(x) = 2^-((exp-127)/2) / sqrt(1.f). The table is for exp odd (x in [1,2) * 4^k) and even. */
	const uint16_t *table = (exp & 1) ? zen2_rsqrt_table_odd : zen2_rsqrt_table_even;
	/* Bochs: exp = 0x7E - ((exp - 0x7F) >> 1) with an arithmetic shift */
	const int32_t sh = ((int32_t)exp - 0x7f) >> 1;
	const uint32_t out_exp = (uint32_t)(0x7e - sh);
	return (out_exp << 23) | ((uint32_t)table[fraction >> 11] << 11);
}

float zen2_approximate_rsqrt(float x) { return bits2f(zen2_approximate_rsqrt_bits(f2bits(x))); }
