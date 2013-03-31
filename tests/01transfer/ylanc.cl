__constant sampler_t sampler_lin = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;

__kernel void filter(float scale, int height, __global float* filter, __read_only image2d_t img_in, __write_only image2d_t img_out) {
	
	int x = get_global_id(0);
	int y = get_global_id(1);

	int left = convert_int(floor(convert_float(y-2) / scale));
	int right = convert_int(ceil(convert_float(y+3) / scale));

	float density = 0.0f, lanc;
	int n = 0;

	float4 sampled = (float4)(0.0f,0.0f,0.0f,0.0f);

	for (int i = 0; i < FILTW; ++i, ++n) {
		lanc = filter[y*FILTW+n];
		density += lanc;
		sampled += lanc * convert_float4(read_imageui(img_in, sampler_lin, (int2)(x, left+i)));
	}

	sampled /= density;

	write_imageui(img_out, (int2)(x, y), convert_uint4(sampled));

}
