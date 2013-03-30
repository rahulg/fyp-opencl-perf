const sampler_t sampler_lin = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;

#ifndef M_PI
#define M_PI 3.141592653589793
#endif
#define LANC_EPSILON (1e-9f)

float lanczos3(float x) {
	float ax = fabs(x);

	return (ax > 3) ? 0.0f :
		((ax < LANC_EPSILON) ? 1.0f :
			3.0f*sinpi(x)*sinpi(x/3.0f)/(M_PI*M_PI*x*x));
}

__kernel void buildcache(float scale, __global float* filter) {

	int x = get_global_id(0);

	float centre = convert_float(x) / scale;
	int left = convert_int(floor(convert_float(x-2) / scale));
	int right = convert_int(ceil(convert_float(x+3) / scale));

	int fw = right-left + 1;
	fw = fw < 6 ? 6 : fw;

	float lanc;

	for (int i = left; i <= right; ++i) {
		lanc = lanczos3(centre - i);
		filter[x*fw+i] = lanc;
	}

}

__kernel void filter_x(float x_scale, __global float* filter, __read_only image2d_t img_in, __write_only image2d_t img_out) {
	
	int
		x = get_global_id(0),
		y = get_global_id(1);

	int left = convert_int(floor(convert_float(x-2) / x_scale));
	int right = convert_int(ceil(convert_float(x+3) / x_scale));

	int fw = right-left + 1;
	fw = fw < 6 ? 6 : fw;

	float density = 0.0f, lanc;

	float4 sampled = (float4)(0.0f,0.0f,0.0f,0.0f);

	for (int i = left; i <= right; ++i) {
		lanc = filter[x*fw+i];
		density += lanc;
		sampled += lanc * convert_float4(read_imageui(img_in, sampler_lin, (int2)(i, y)));
	}

	sampled /= density;

	write_imageui(img_out, (int2)(x, y), convert_uint4(sampled));

}

__kernel void filter_y(float y_scale, __global float* filter, __read_only image2d_t img_in, __write_only image2d_t img_out) {
	
	int
		x = get_global_id(0),
		y = get_global_id(1);

	int left = convert_int(floor(convert_float(y-2) / y_scale));
	int right = convert_int(ceil(convert_float(y+3) / y_scale));

	int fw = right-left + 1;
	fw = fw < 6 ? 6 : fw;

	float density = 0.0f, lanc;

	float4 sampled = (float4)(0,0,0,0);

	for (int i = left; i < right; ++i) {
		lanc = filter[y*fw+i];
		density += lanc;
		sampled += lanc * convert_float4(read_imageui(img_in, sampler_lin, (int2)(x, i)));
	}

	sampled /= density;

	write_imageui(img_out, (int2)(x, y), convert_uint4(sampled));

}
