#ifndef M_PI
#define M_PI 3.141592653589793
#endif
#define LANC_EPSILON (1e-9f)

inline float lanczos3(float x) {
	float ax = fabs(x);

	return (ax >= 3.0f) ? 0.0f :
		((ax < LANC_EPSILON) ? 1.0f :
			3.0f*sinpi(x)*sinpi(x/3.0f)/(M_PI*M_PI*x*x));
}

__kernel void testkern(__global float* infilter, __global float* filter) {

	int x = get_global_id(0);

	float temp = 0.0f;

	for (int i = 0; i < 6000; ++i) {
		temp += lanczos3(3 - i) * lanczos3(3 - i) * infilter[x];
	}

	filter[x] = temp;

}
