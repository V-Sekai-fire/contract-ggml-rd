#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
int main(void) {
	FILE *f = fopen("sc_in.bin", "rb"); fseek(f, 0, SEEK_END); long n = ftell(f) / 4; rewind(f);
	float *in = malloc(n * 4), *rs = malloc(n * 4), *rc = malloc(n * 4);
	fread(in, 4, n, f); fclose(f);
	f = fopen("sc_sin.bin", "rb"); fread(rs, 4, n, f); fclose(f);
	f = fopen("sc_cos.bin", "rb"); fread(rc, 4, n, f); fclose(f);
	long b[4] = {0, 0, 0, 0};
	for (long i = 0; i < n; ++i) {
		float cand[4] = { sinf(in[i]), (float)sin((double)in[i]), cosf(in[i]), (float)cos((double)in[i]) };
		float want[4] = { rs[i], rs[i], rc[i], rc[i] };
		for (int k = 0; k < 4; ++k) { uint32_t a, c; memcpy(&a, &cand[k], 4); memcpy(&c, &want[k], 4); if (a != c) b[k]++; }
	}
	printf("of %ld inputs differ from XLA: sinf %ld | (float)sin(double) %ld | cosf %ld | (float)cos(double) %ld\n", n, b[0], b[1], b[2], b[3]);
	return 0;
}
