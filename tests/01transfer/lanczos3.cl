const sampler_t sampler_lin = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;

__kernel void filter_x(float x_scale, __global float* filter, __read_only image2d_t img_in, __write_only image2d_t img_out) {
	
	int
		x = get_global_id(0),
		y = get_global_id(1);

	int
		in_x;

	int2 c = (int2)(x, y);

	float4 sampled = (float4)(0,0,0,0);

	for (int i = -2; i <= 3; ++i) {
		in_x = convert_int(floor(convert_float(x+i) / x_scale));
		sampled += filter[8*x+i+2] * convert_float4(read_imageui(img_in, sampler_lin, (int2)(in_x, y)));
	}

	write_imageui(img_out, c, convert_uint4(sampled));

}

__kernel void filter_y(float y_scale, __global float* filter, __read_only image2d_t img_in, __write_only image2d_t img_out) {
	
	int
		x = get_global_id(0),
		y = get_global_id(1);

	int
		in_y;

	int2 c = (int2)(x, y);

	float4 sampled = (float4)(0,0,0,0);

	for (int i = -2; i <= 3; ++i) {
		in_y = convert_int(floor(convert_float(y+i) / y_scale));
		sampled += filter[8*x+i+2] * convert_float4(read_imageui(img_in, sampler_lin, (int2)(x, in_y)));
	}

	write_imageui(img_out, c, convert_uint4(sampled));

}
