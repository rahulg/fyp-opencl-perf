const sampler_t sampler_lin = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define LANC_EPSILON (1e-9f)


inline float lanczos3(float x) {
	float ax = fabs(x);

	float ret =
	(ax > 3) ? 0.0f :
		((ax < LANC_EPSILON) ? 1.0f :
			3.0f*native_sin(M_PI*x)*native_sin(M_PI*x/3.0f)/(M_PI*M_PI*x*x));

	return ret;
}

__kernel void pregen(float x_scale, float y_scale, __global float* filter_x, __global float* filter_y) {
	float tx = 0.0f, ty = 0.0f;
	for (int i = 0; i < 7; ++i) {
		tx += filter_x[i] = lanczos3(convert_float(i-3) * x_scale);
		ty += filter_y[i] = lanczos3(convert_float(i-3) * y_scale);
	}
	for (int i = 0; i < 7; ++i) {
		filter_x[i] /= tx;
		filter_y[i] /= ty;
	}
}

__kernel void filter_x(float x_scale, __global float* filter, __read_only image2d_t img_in, __write_only image2d_t img_out) {
	
	int
		x = get_global_id(0),
		y = get_global_id(1);

	float
		in_x,
		in_y = convert_float(y);

	int2 c = (int2)(x, y);

	float4 sampled = (float4)(0,0,0,0);

	for (int i = -3; i <= 3; ++i) {
		in_x = convert_float(x+i) / x_scale;
		sampled += filter[i+3] * convert_float4(read_imageui(img_in, sampler_lin, (float2)(in_x, in_y)));
	}

	write_imageui(img_out, c, convert_uint4(sampled));

}

__kernel void filter_y(float y_scale, __global float* filter, __read_only image2d_t img_in, __write_only image2d_t img_out) {
	
	int
		x = get_global_id(0),
		y = get_global_id(1);

	float
		in_x = convert_float(x),
		in_y;

	int2 c = (int2)(x, y);

	float4 sampled = (float4)(0,0,0,0);

	for (int i = -3; i <= 3; ++i) {
		in_y = convert_float(y+i) / y_scale;
		sampled += filter[i+3] * convert_float4(read_imageui(img_in, sampler_lin, (float2)(in_x, in_y)));
	}

	write_imageui(img_out, c, convert_uint4(sampled));

}
