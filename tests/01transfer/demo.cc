#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#include <fcntl.h>

#include <cmath>
#include <algorithm>

#include "derpcl/cl.h"
#include "timing.h"

using namespace derpcl;
using namespace std;

typedef struct {
	Image host_s, host_d, dev_s, dev_x, dev_d;
	uint8_t *src_p, *dest_p;
	cl_event eread, escale, efx, efy, ewrite;
} plane_t;

float lanczos3(float x);

int main(int argc, char const *argv[])
{

	bool pipe_out = false;

_X_TIMER_SETUP

	if (argc < 5)
	{
		cerr << "Usage: filter <input_file> <output_file>"\
			" <width> <height>" << endl;
		return 1;
	}

	string input_name = string(argv[1]);
	string output_name = string(argv[2]);
	uint32_t out_width = atoi(argv[3]);
	uint32_t out_height = atoi(argv[4]);

	ofstream dest_m;
	int src_fd, dest_fd;

	src_fd = open(input_name.c_str(), O_RDONLY);
	ifstream source_m(input_name + ".meta", ios::in);

	if (src_fd == -1)
	{
		perror("Source file error");
		throw exception();
	}

	if (output_name == "-")
	{
		dest_fd = STDOUT_FILENO;
		pipe_out = true;
	}
	else
	{
		dest_fd = open(output_name.c_str(), O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP);
		if (dest_fd == -1)
		{
			perror("Destination file error");
			throw exception();
		}
		dest_m.open(output_name + ".meta", ios::out | ios::trunc);
	}

	uint32_t in_width, in_height;
	string
		flag_width = "WIDTH=",
		flag_height = "HEIGHT=";

	for (int i = 0; i < 2; ++i)
	{
		size_t pos;
		char temp_ca[256];

		source_m.getline(temp_ca, 256);
		string temp_s = string(temp_ca);

		pos = temp_s.find(flag_width);
		if (pos != string::npos)
		{
			istringstream str_stream(temp_s.substr(pos+6));
			str_stream >> in_width;
		}

		pos = temp_s.find(flag_height);
		if (pos != string::npos)
		{
			istringstream str_stream(temp_s.substr(pos+7));
			str_stream >> in_height;
		}
	}

	size_t
		in_lum_sz = in_width * in_height,
		in_chr_sz = (in_width/2) * (in_height/2),
		out_lum_sz = out_width * out_height,
		out_chr_sz = (out_width/2) * (out_height/2);

	cl_float
		x_scale = (float)out_width / (float)in_width,
		y_scale = (float)out_height / (float)in_height;

	struct stat st_buf;
	stat(input_name.c_str(), &st_buf);

	uint64_t n_frames = st_buf.st_size / (in_lum_sz + (2 * in_chr_sz));

	if (!pipe_out)
	{
		dest_m << flag_width << out_width << endl;
		dest_m << flag_height << out_height << endl;
	}

	cerr << "[i] width: " << in_width << endl
	     << "[i] height: " << in_height << endl
	     << "[i] lum_sz: " << in_lum_sz
	     << " | chr_sz: " << in_chr_sz << endl
	     << "[-] frames: " << n_frames << endl
	     << "[-] xscale: " << x_scale
	     << " | yscale: " << y_scale << endl
	     << "[o] width: " << out_width << endl
	     << "[o] height: " << out_height << endl
	     << "[o] lum_sz: " << out_lum_sz
	     << " | chr_sz: " << out_chr_sz << endl;

	Environment env(DeviceType::GPU);

	Program filters(env, "lanczos3.cl", false);

	Kernel
		fx_y(filters, "filter_x"),
		fx_u(filters, "filter_x"),
		fx_v(filters, "filter_x"),
		fy_y(filters, "filter_y"),
		fy_u(filters, "filter_y"),
		fy_v(filters, "filter_y");

#define HOST_IMG(width,height) Image(env, MemoryType::PinnedReadWrite, Channels::Single, PixelFormat::Unsigned8, {(width), (height)})
#define DEV_IMG(width,height) Image(env, MemoryType::ReadWrite, Channels::Single, PixelFormat::Unsigned8, {(width), (height)})
#define DEV_RIMG(width,height) Image(env, MemoryType::ReadOnly, Channels::Single, PixelFormat::Unsigned8, {(width), (height)})
#define DEV_WIMG(width,height) Image(env, MemoryType::WriteOnly, Channels::Single, PixelFormat::Unsigned8, {(width), (height)})
#define FILTER(size) Buffer<cl_float>(env, MemoryType::ReadWrite, size*8)

#define LUM_PLANE(iw,ih,ow,oh) \
HOST_IMG(iw, ih), \
HOST_IMG(ow, oh), \
DEV_RIMG(iw, ih), \
DEV_IMG(ow, ih), \
DEV_WIMG(ow, oh)
#define CHR_PLANE(iw,ih,ow,oh) LUM_PLANE(iw/2, ih/2, ow/2, oh/2)

	plane_t
	y = {
		LUM_PLANE(in_width, in_height, out_width, out_height)
	},
	u = {
		CHR_PLANE(in_width, in_height, out_width, out_height)
	},
	v = {
		CHR_PLANE(in_width, in_height, out_width, out_height)
	};


	double ttl_time = 0.0;
	uint64_t stime, etime;

	try {

#define MAP_HPTRS(plane) \
plane.host_s.map(MapMode::Write); \
plane.host_d.map(MapMode::Read);

		MAP_HPTRS(y);
		MAP_HPTRS(u);
		MAP_HPTRS(v);

#define SET_HPTRS(plane) \
plane.src_p = static_cast<uint8_t*>(plane.host_s.data()); \
plane.dest_p = static_cast<uint8_t*>(plane.host_d.data());

		SET_HPTRS(y);
		SET_HPTRS(u);
		SET_HPTRS(v);

#define SET_FX_ARG(kern,plane) \
kern.setArgument(0, x_scale); \
kern.setArgumentImage(1, (plane).dev_s); \
kern.setArgumentImage(2, (plane).dev_x);

		SET_FX_ARG(fx_y, y);
		SET_FX_ARG(fx_u, u);
		SET_FX_ARG(fx_v, v);

#define SET_FY_ARG(kern,plane) \
kern.setArgument(0, y_scale); \
kern.setArgumentImage(1, (plane).dev_x); \
kern.setArgumentImage(2, (plane).dev_d);

		SET_FY_ARG(fy_y, y);
		SET_FY_ARG(fy_u, u);
		SET_FY_ARG(fy_v, v);

		stime = _x_time();
		ssize_t rd_count;

		for (uint64_t i = 0; i < n_frames; ++i)
		{
			fprintf(stderr, "Scaling frame %lld\r", i);

			// Read Y
			rd_count = read(src_fd, y.src_p, in_lum_sz);
			if (rd_count != in_lum_sz)
			{
				cerr << "[Y] Read failed" << endl;
			}

			y.ewrite = y.dev_s.queueWrite(y.src_p);
			y.efx = fx_y.run(out_width, in_height, y.ewrite);
			y.efy = fy_y.run(out_width, out_height, y.efx);
			y.eread = y.dev_d.queueRead(y.dest_p, y.efy);

			// Read U
			rd_count = read(src_fd, u.src_p, in_chr_sz);
			if (rd_count != in_chr_sz)
			{
				cerr << "[U] Read failed" << endl;
			}

			u.ewrite = u.dev_s.queueWrite(u.src_p);
			u.efx = fx_u.run(out_width/2, in_height/2, u.ewrite);
			u.efy = fy_u.run(out_width/2, out_height/2, u.efx);
			u.eread = u.dev_d.queueRead(u.dest_p, u.efy);

			// Read V
			rd_count = read(src_fd, v.src_p, in_chr_sz);
			if (rd_count != in_chr_sz)
			{
				cerr << "[V] Read failed" << endl;
			}

			v.ewrite = v.dev_s.queueWrite(v.src_p);
			v.efx = fx_v.run(out_width/2, in_height/2, v.ewrite);
			v.efy = fy_v.run(out_width/2, out_height/2, v.efx);
			v.eread = v.dev_d.queueRead(v.dest_p, v.efy);

			fprintf(stderr, "Scaling frame %lld\r", i+1);

			// Write Y
			clWaitForEvents(1, &y.eread);
			write(dest_fd, y.dest_p, out_lum_sz);

			// Write U
			clWaitForEvents(1, &u.eread);
			write(dest_fd, u.dest_p, out_chr_sz);

			// Write V
			clWaitForEvents(1, &v.eread);
			write(dest_fd, v.dest_p, out_chr_sz);

		}

		etime = _x_time();
		ttl_time = (double)(etime-stime)/1000000ll;

		cerr << "[-] Time: " <<  ttl_time << " ms" << endl;
		cerr << "[-] Time per frame: " <<  ttl_time / n_frames << " ms" << endl;
		cerr << "[-] FPS: " <<  n_frames * 1000 / ttl_time << endl;

#define UNMAP_HPTRS(plane) \
plane.host_s.unmap(); \
plane.host_d.unmap();

		UNMAP_HPTRS(y);
		UNMAP_HPTRS(u);
		UNMAP_HPTRS(v);

		close(src_fd);
		if (!pipe_out)
		{
			close(dest_fd);
		}
		source_m.close();
		dest_m.close();
	} catch (string s) {
		cerr << s << endl;
	}

_X_TIMER_TEARDOWN

	return 0;
}

#define LANC_EPSILON (1e-9f)

float lanczos3(float x) {
	float ax = fabs(x);

	return (ax > 3) ? 0.0f :
		((ax < LANC_EPSILON) ? 1.0f :
			3.0f*sin(M_PI*x)*sin(M_PI*x/3.0f)/(M_PI*M_PI*x*x));
}
