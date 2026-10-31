#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "zen2_rsqrt.c"
int main(void) {
	uint64_t bad = 0, nan_payload = 0; uint32_t first = 0; int have = 0;
	for (uint64_t i = 0; i < (1ull << 32); ++i) {
		uint32_t in = (uint32_t)i;
		float f; memcpy(&f, &in, 4);
		float r = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(f)));
		uint32_t want; memcpy(&want, &r, 4);
		uint32_t got = zen2_approximate_rsqrt_bits(in);
		if (got != want) {
			int bothnan = ((want & 0x7fffffff) > 0x7f800000) && ((got & 0x7fffffff) > 0x7f800000);
			if (bothnan) { nan_payload++; continue; }
			if (!have) { first = in; have = 1; printf("first miss in=%08x want=%08x got=%08x\n", in, want, got); }
			bad++;
		}
	}
	printf("zen2_approximate_rsqrt vs RSQRTSS over 2^32 inputs: %llu differ (%llu NaN payloads ignored)\n", (unsigned long long)bad, (unsigned long long)nan_payload);
	return bad != 0;
}
