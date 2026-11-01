#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
int main(void) {
	FILE *f = fopen("pw_a.bin", "rb"); fseek(f, 0, SEEK_END); long n = ftell(f) / 4; rewind(f);
	float *a = malloc(n * 4), *b = malloc(n * 4), *o = malloc(n * 4);
	fread(a, 4, n, f); fclose(f);
	f = fopen("pw_b.bin", "rb"); fread(b, 4, n, f); fclose(f);
	f = fopen("pw_out.bin", "rb"); fread(o, 4, n, f); fclose(f);
	long d1 = 0, d2 = 0, d3 = 0;
	for (long i = 0; i < n; ++i) {
		float c1 = powf(a[i], b[i]), c2 = (float)pow((double)a[i], (double)b[i]), c3 = expf(b[i] * logf(a[i]));
		uint32_t x, y; memcpy(&y, &o[i], 4);
		memcpy(&x, &c1, 4); d1 += x != y; memcpy(&x, &c2, 4); d2 += x != y; memcpy(&x, &c3, 4); d3 += x != y;
	}
	printf("of %ld inputs differ from XLA: powf %ld | (float)pow(double) %ld | expf(b*logf(a)) %ld\n", n, d1, d2, d3);
	return 0;
}
