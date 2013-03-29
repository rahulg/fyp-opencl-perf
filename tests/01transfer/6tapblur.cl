const sampler_t sampler_near = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;

__kernel void rescale(float x_scale, float y_scale, __read_only image2d_t img_in, __write_only image2d_t img_out)
{
    
	int
		out_x = get_global_id(0),
		out_y = get_global_id(1);

	float
		in_x = x_scale * (float)(out_x),
		in_y = y_scale * (float)(out_y);

	write_imageui(img_out, (int2)(out_x, out_y), convert_uint4(read_imageui(img_in, sampler_near, (float2)(in_x, in_y))));

}

__kernel void filter_x(__read_only image2d_t img_in, __write_only image2d_t img_out)
{
    
	int
		x = get_global_id(0),
		y = get_global_id(1);

	const float filter[6] = {
		0.2, 0.2, 0.2, 0.2, 0.1, 0.1
	};	

	int2 c = (int2)(x, y);

	float4 sampled = (float4)(0,0,0,0);

	for (int i = 0; i < 6; ++i) {
		sampled += filter[i] * convert_float4(read_imageui(img_in, sampler_near, (float2)(x+i, y)));
	}

	write_imageui(img_out, c, convert_uint4(sampled));

}

__kernel void filter_y(__read_only image2d_t img_in, __write_only image2d_t img_out)
{
    
	int
		x = get_global_id(0),
		y = get_global_id(1);

	const float filter[6] = {
		0.2, 0.2, 0.2, 0.2, 0.1, 0.1
	};	

	int2 c = (int2)(x, y);

	float4 sampled = (float4)(0,0,0,0);

	for (int i = 0; i < 6; ++i) {
		sampled += filter[i] * convert_float4(read_imageui(img_in, sampler_near, (float2)(x, y+i)));
	}

	write_imageui(img_out, c, convert_uint4(sampled));

}
