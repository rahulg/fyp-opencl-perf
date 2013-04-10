const sampler_t sampler_nn = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;

__kernel void convolve(__global ushort* filter, __read_only image2d_t in, __write_only image2d_t out) {
	int2 c = (int2)(get_global_id(0), get_global_id(1));
	uint4 sampled = (uint4)(0,0,0,0);
	for (int i = 0; i < FWIDTH; ++i) {
		sampled += filter[i] * read_imageui(in, sampler_nn,(int2)(c.x+i,c.y));
	}
	write_imageui(out, c, sampled);
}

__kernel void convolve4(__global ushort4* filter0, __global ushort4* filter1, __global ushort4* filter2, __global ushort4* filter3, __read_only image2d_t in, __write_only image2d_t out) {
	int2 c = (int2)(get_global_id(0), get_global_id(1));
	uint4 sampled0 = (uint4)(0,0,0,0);
	uint4 sampled1 = (uint4)(0,0,0,0);
	uint4 sampled2 = (uint4)(0,0,0,0);
	uint4 sampled3 = (uint4)(0,0,0,0);
	uint4 ipix, opix;
	for (int i = 0; i < FWIDTH/4; ++i) {
		ipix = read_imageui(in, sampler_nn,(int2)(c.x+i,c.y));
		sampled0 += convert_uint4(filter0[i]) * ipix;
		sampled1 += convert_uint4(filter1[i]) * ipix;
		sampled2 += convert_uint4(filter2[i]) * ipix;
		sampled3 += convert_uint4(filter3[i]) * ipix;
	}
	opix.x = sampled0.x + sampled0.y + sampled0.z + sampled0.w;
	opix.y = sampled1.x + sampled1.y + sampled1.z + sampled1.w;
	opix.z = sampled2.x + sampled2.y + sampled2.z + sampled2.w;
	opix.w = sampled3.x + sampled3.y + sampled3.z + sampled3.w;
	write_imageui(out, c, opix);
}
