const sampler_t sampler_nn = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;

__kernel void convolve(__global ushort* filter, __read_only image2d_t in, __write_only image2d_t out) {
	int2 c = (int2)(get_global_id(0), get_global_id(1));
	uint4 sampled = (uint4)(0,0,0,0);
	for (int i = 0; i < FWIDTH; ++i) {
		sampled += filter[i] * read_imageui(in, sampler_nn,c);
	}
	write_imageui(out, c, sampled);
}
