__kernel void compb(__global uchar4* in, __global uchar4* out) {

	int x = get_global_id(0);
	int y = get_global_id(1);

	int w = get_global_size(0);

	uchar4 temp = (uchar4)(0.0f,0.0f,0.0f,0.0f);

	for (int i = 0; i < 45; ++i) {
		temp += convert_uchar4(in[(y+i)*w+x]);
	}

	out[y*w+x] = temp;

}

__constant sampler_t sampler_lin = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;
__kernel void compp(__read_only image2d_t in, __write_only image2d_t out) {

	int x = get_global_id(0);
	int y = get_global_id(1);

	uchar4 temp = (uchar4)(0.0f,0.0f,0.0f,0.0f);

	for (int i = 0; i < 45; ++i) {
		temp += convert_uchar4(read_imageui(in, sampler_lin, (int2)(x, y+i)));
	}

	write_imageui(out, (int2)(x,y), convert_uint4(temp));

}
