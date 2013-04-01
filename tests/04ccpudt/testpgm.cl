__constant sampler_t sampler_lin = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;
__kernel void compi(__read_only image2d_t in, __write_only image2d_t out) {

	int x = get_global_id(0);
	int y = get_global_id(1);

	uchar4 temp = (uchar4)(0,0,0,0);
	uchar4 mul = (uchar4)(1,2,1,2);

	for (int i = 0; i < 45; ++i) {
		temp += mul * convert_uchar4(read_imageui(in, sampler_lin, (int2)(x, y+i)));
	}

	temp /= 2;

	write_imageui(out, (int2)(x,y), convert_uint4(temp));

}

__kernel void compf(__read_only image2d_t in, __write_only image2d_t out) {

	int x = get_global_id(0);
	int y = get_global_id(1);

	float4 temp = (float4)(0.0f,0.0f,0.0f,0.0f);
	float4 mul = (float4)(0.11f,0.22f,0.33f,0.44f);

	for (int i = 0; i < 45; ++i) {
		temp += mul * convert_float4(read_imageui(in, sampler_lin, (int2)(x, y+i)));
	}

	temp /= 2.0f;

	write_imageui(out, (int2)(x,y), convert_uint4(temp));

}
