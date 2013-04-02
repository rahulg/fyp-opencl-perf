__constant sampler_t sampler_lin = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;

__kernel void filter(float scale, int height, __global short* filter, __read_only image2d_t img_in, __write_only image2d_t img_out) {
	
	int x = get_global_id(0);
	int y = get_global_id(1);

	int left = convert_int(floor(convert_float(x-2) / scale));
	int right = convert_int(ceil(convert_float(x+3) / scale));

	int density = 0, lanc;
	int n = 0;

	int4 sampled = (int4)(0,0,0,0);

	for (int i = 0; i < FILTW; ++i, ++n) {
		lanc = convert_int(filter[x*FILTW+n]);
		density += lanc;
		sampled += lanc * convert_int4(read_imageui(img_in, sampler_lin, (int2)(left+i, y)));
	}

	sampled /= density;

	sampled = clamp(sampled, (int4)(0,0,0,0), (int4)(255,255,255,255));

	write_imageui(img_out, (int2)(x, y), convert_uint4(sampled));

}
