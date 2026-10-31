// The exact arithmetic of the GPU kernel rsqrt_zen2, in C, against the host model of XLA CPU's rsqrt on Zen 2.
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "xla_rsqrt.c"
#include "zen2_words.h"

static uint32_t kernel_bits(uint32_t bits) {
	float x = bits2f(bits);
	if (x != x) return bits;
	uint32_t ex = (bits >> 23) & 255u;
	if (bits >> 31) return ex == 0 ? 0xff800000u : 0xffc00000u;
	if (ex == 0) return 0x7f800000u;
	if (bits == 0x7f800000u) return 0u;
	uint32_t idx = (bits & 0x7fffffu) >> 11;
	uint32_t par = (ex & 1u) ? 0u : 1u;
	uint32_t n = par * 4096u + idx;
	int res = (int)((zen2_words[n >> 4] >> (2u * (n & 15u))) & 3u) - 1;
	float scale = par ? 2.0f : 1.0f;
	float xm = (1.0f + ((float)idx * 2.0f + 1.0f) / 8192.0f) * scale;
	float v = 8192.0f / sqrtf(xm);
	int y = (int)floorf(v + 0.5f) + res;
	int sh = ((int)ex - 127) >> 1;
	uint32_t eb = ((uint32_t)(0x7e - sh) << 23) | ((uint32_t)(y - 4096) << 11);
	float r = bits2f(eb);
	float r2 = x * r;
	float half_r = r * -0.5f;
	float h = fmaf(r2, r, -1.0f);
	r = fmaf(half_r, h, r);
	return f2bits(r);
}

int main(void) {
	uint64_t bad = 0; int shown = 0;
	for (uint64_t i = 0; i < (1ull << 32); ++i) {
		uint32_t in = (uint32_t)i;
		float want = xla_rsqrt1(bits2f(in), 0, 1, 1);
		uint32_t w = f2bits(want), g = kernel_bits(in);
		if (w == g) continue;
		if (((w & 0x7fffffffu) > 0x7f800000u) && ((g & 0x7fffffffu) > 0x7f800000u)) continue;
		if (shown++ < 4) printf("miss in=%08x xla=%08x kernel=%08x\n", in, w, g);
		bad++;
	}
	printf("kernel arithmetic vs XLA model over 2^32 inputs: %llu differ\n", (unsigned long long)bad);
	return bad != 0;
}
