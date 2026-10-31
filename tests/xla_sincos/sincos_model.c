#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
static float rnd(float x) { return nearbyintf(x); }
static float model(float x, int cosine, int fuse) {
	const float pi = 3.1415927f, inv2pi = 0.15915494f, pih_hi = 1.5707963705e+00f, pih_lo = -4.3711388e-08f;
	const float t_hi = 6.28125f, t_lo = 1.9353072e-3f;
	const float a3 = -1.3314664364e-01f, a5 = 3.2340581529e-03f, b2 = 3.3519912511e-02f, b4 = 4.8770775902e-04f;
	float n = rnd(x * inv2pi);
	x = fuse ? fmaf(-n, t_hi, x) : x - n * t_hi;
	x = fuse ? fmaf(-n, t_lo, x) : x - n * t_lo;
	if (cosine) x = (pih_hi - x) + pih_lo;
	x = fminf(x, pi - x);
	x = fmaxf(x, (-pi) - x);
	x = fminf(x, pi - x);
	float x2 = x * x;
	float p = fuse ? fmaf(x2, a5, a3) : x2 * a5 + a3;
	p = fuse ? fmaf(x2, p, 1.0f) : x2 * p + 1.0f;
	p = x * p;
	float q = fuse ? fmaf(x2, b4, b2) : x2 * b4 + b2;
	q = fuse ? fmaf(x2, q, 1.0f) : x2 * q + 1.0f;
	return p / q;
}
int main(void) {
	FILE *f = fopen("sc_in.bin", "rb"); fseek(f, 0, SEEK_END); long n = ftell(f) / 4; rewind(f);
	float *in = malloc(n * 4), *rs = malloc(n * 4), *rc = malloc(n * 4);
	fread(in, 4, n, f); fclose(f);
	f = fopen("sc_sin.bin", "rb"); fread(rs, 4, n, f); fclose(f);
	f = fopen("sc_cos.bin", "rb"); fread(rc, 4, n, f); fclose(f);
	for (int fuse = 0; fuse < 2; ++fuse) for (int c = 0; c < 2; ++c) {
		long bad = 0, bad_small = 0, ns = 0; int first = -1;
		for (long i = 0; i < n; ++i) {
			float m = model(in[i], c, fuse), want = c ? rc[i] : rs[i];
			uint32_t a, b; memcpy(&a, &m, 4); memcpy(&b, &want, 4);
			if (a != b && !(m != m && want != want)) { bad++; if (first < 0) first = (int)i; if (fabsf(in[i]) < 100) bad_small++; }
		}
		printf("%s fuse=%d: %ld of %ld differ (|x|<100: %ld)", c ? "cos" : "sin", fuse, bad, n, bad_small);
		if (first >= 0) { float m = model(in[first], c, fuse); printf("  e.g. x=%g model=%.9g xla=%.9g", in[first], m, c ? rc[first] : rs[first]); }
		printf("\n");
	}
	return 0;
}
