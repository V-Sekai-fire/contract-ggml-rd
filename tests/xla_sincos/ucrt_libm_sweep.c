#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ucrt_libm.c"
int main(void) {
	unsigned long long bad[2] = {0, 0}, shown[2] = {0, 0}, big[2] = {0, 0};
	for (unsigned long long i = 0; i < (1ull << 32); ++i) {
		uint32_t in = (uint32_t)i; float x; memcpy(&x, &in, 4);
		float w[2] = { sinf(x), cosf(x) }, g[2] = { ucrt_sinf(x), ucrt_cosf(x) };
		for (int k = 0; k < 2; ++k) {
			uint32_t a, b; memcpy(&a, &w[k], 4); memcpy(&b, &g[k], 4);
			if (a == b || (w[k] != w[k] && g[k] != g[k])) continue;
			bad[k]++; if ((in & 0x7fffffffu) >= 0x4b800456u) big[k]++;
			if (shown[k]++ < 4) printf("%s miss x=%08x want=%08x got=%08x\n", k ? "cos" : "sin", in, a, b);
		}
	}
	printf("ucrt_sinf vs sinf over 2^32 inputs: %llu differ (%llu with |x| >= 2^24)\n", bad[0], big[0]);
	printf("ucrt_cosf vs cosf over 2^32 inputs: %llu differ (%llu with |x| >= 2^24)\n", bad[1], big[1]);
	return (bad[0] | bad[1]) != 0;
}
