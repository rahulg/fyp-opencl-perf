#ifndef M_PI
#define M_PI 3.141592653589793f
#endif
#define LANC_EPSILON (1e-9f)

inline float lanczos3(float x) {
	float ax = fabs(x);

	return (ax >= 3.0f) ? 0.0f :
		((ax < LANC_EPSILON) ? 1.0f :
			3.0f*sinpi(x)*sinpi(x/3.0f)/(M_PI*M_PI*x*x));
}

__kernel void cache(float scale, __global short* filter) {

	int x = get_global_id(0);

	float centre = convert_float(x) / scale;
	int left = convert_int(floor(convert_float(x-2) / scale));
	int right = convert_int(ceil(convert_float(x+3) / scale));

	float ffilter[FILTW], fweight=0.0f;
	int n = 0;

	for (int i = left; i <= right; ++i, ++n) {
		ffilter[n] = lanczos3(centre - i);
		fweight += ffilter[n];
	}

	// normalise floatfilter
	for (int i = 0; i < FILTW; ++i) {
		ffilter[i] /= fweight;
	}

	for (int i = 0; i < FILTW; ++i) {
		filter[x*FILTW+i] = convert_short(ffilter[i] * 32767.0f);
	}

}
