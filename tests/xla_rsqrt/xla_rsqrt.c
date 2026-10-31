// Host models of XLA CPU's f32 rsqrt (xla/codegen/intrinsic/rsqrt.cc): hardware estimate + Newton-Raphson.
// Built with -ffp-contract=off. est: 0 = rsqrt_ss (SSE), 1 = rsqrtps 8-wide (AVX), 2 = rsqrt14 (AVX512).
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static float bits2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t f2bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static float estimate(float x, int est) {
	if (est == 0) return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x)));
	if (est == 1) return _mm_cvtss_f32(_mm_rsqrt_ps(_mm_set1_ps(x)));
#ifdef __AVX512F__
	return _mm_cvtss_f32(_mm512_castps512_ps128(_mm512_rsqrt14_ps(_mm512_set1_ps(x))));
#else
	return 0.0f;
#endif
}

/* fuse: 1 = the two PMAdd per step are fused (fmaf), 0 = separate */
static float xla_rsqrt1(float x, int est, int steps, int fuse) {
	float y = estimate(x, est);
	float r = y;
	for (int s = 0; s < steps; ++s) {
		float r2 = x * r;
		float half_r = r * -0.5f;
		float h = fuse ? fmaf(r2, r, -1.0f) : r2 * r + -1.0f;
		r = fuse ? fmaf(half_r, h, r) : half_r * h + r;
	}
	/* select(x < FLT_MIN || x == inf, y_approx, refined) */
	if (x < bits2f(0x00800000u) || x == INFINITY) return y;
	return r;
}

__declspec(dllexport) void rsqrt_many(const uint32_t *in, uint32_t *out, long n, int est, int steps, int fuse) {
	for (long i = 0; i < n; ++i) out[i] = f2bits(xla_rsqrt1(bits2f(in[i]), est, steps, fuse));
}
__declspec(dllexport) int has_avx512(void) {
#ifdef __AVX512F__
	return 1;
#else
	return 0;
#endif
}
