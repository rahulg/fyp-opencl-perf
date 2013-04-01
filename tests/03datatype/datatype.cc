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

#include "config.h"

#define CLEV_TIME(ev,dbl) \
clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL); \
clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL); \
dbl += (double)(end - start)*(double)(1e-06);

int main(int argc, char const *argv[])
{

_X_TIMER_SETUP

	size_t bufsz = WIDTH * HEIGHT;
	size_t iter = ITER;

	
	try {

		Environment env(DeviceType::GPU);
		Program pg(env, "testpgm.cl");

		// IMAGE
		Kernel kerni(pg, "compi");
		Kernel kernf(pg, "compf");
		Image img0(env, MemoryType::ReadOnly, Channels::RGBA, PixelFormat::Unsigned8, {WIDTH, HEIGHT*2});
		Image img1(env, MemoryType::WriteOnly, Channels::RGBA, PixelFormat::Unsigned8, {WIDTH, HEIGHT*2});

		kerni.setArgumentImage(0, img0);
		kerni.setArgumentImage(1, img1);
		kernf.setArgumentImage(0, img0);
		kernf.setArgumentImage(1, img1);

		Image ih0(env, MemoryType::PinnedReadWrite, Channels::RGBA, PixelFormat::Unsigned8, {WIDTH, HEIGHT*2});
		Image ih1(env, MemoryType::PinnedReadWrite, Channels::RGBA, PixelFormat::Unsigned8, {WIDTH, HEIGHT*2});
		ih0.map(MapMode::Write);
		ih1.map(MapMode::Read);
		cl_uchar *ihost0 = static_cast<cl_uchar*>(ih0.data());
		cl_uchar *ihost1 = static_cast<cl_uchar*>(ih1.data());

		for (int i = 0; i < bufsz * 8; ++i) {
			ihost0[i] = i % 128;
		}

		cl_ulong start, end;
		double wrtime=0.0, runtime=0.0, retime=0.0;
		uint64_t stime, etime;
		size_t ndims[2] = {WIDTH,HEIGHT};

		size_t origin[3] = {0,0,0};
		size_t endpoint[3] = {WIDTH,HEIGHT,1};
		cl_event wre, rune, ree;

		/***************INT****************/
		clEnqueueWriteImage(env.txQueue(), img0.block(), CL_TRUE, origin, endpoint, 0, 0, ihost0, 0, NULL, &wre);
		
		for (int i = 0; i < WARMUPITER; ++i)
		{
			fprintf(stderr, "\rIWARM %d", i);
			clEnqueueNDRangeKernel(env.queue(), kerni.kernel(), 2, NULL, ndims, NULL, 0, NULL, &rune);
			clWaitForEvents(1, &rune);
		}
		clEnqueueReadImage(env.txQueue(), img1.block(), CL_TRUE, origin, endpoint, 0, 0, ihost1, 0, NULL, &ree);
		clWaitForEvents(1, &ree);

		cerr << "\nint kernel:" << endl;

		stime = _x_time();
		clEnqueueWriteImage(env.txQueue(), img0.block(), CL_TRUE, origin, endpoint, 0, 0, ihost0, 0, NULL, &wre);

		for (int i = 0; i < ITER; ++i)
		{
			fprintf(stderr, "\rITER %d", i);
			clEnqueueNDRangeKernel(env.queue(), kerni.kernel(), 2, NULL, ndims, NULL, 0, NULL, &rune);
			clWaitForEvents(1, &rune);

			CLEV_TIME(rune, runtime);
		}
		clEnqueueReadImage(env.txQueue(), img1.block(), CL_TRUE, origin, endpoint, 0, 0, ihost1, 0, NULL, &ree);
		clWaitForEvents(1, &ree);

		etime = _x_time();
		fprintf(stderr, "\n");

		cerr << "RUN: " << runtime / ITER << endl;
		cerr << "CTM: " << (double)(etime-stime)/1000000ll / ITER << endl;
		cerr << "FPS: " << ITER * 1000 / runtime << endl;

		/***************FLOAT****************/
		clEnqueueWriteImage(env.txQueue(), img0.block(), CL_TRUE, origin, endpoint, 0, 0, ihost0, 0, NULL, &wre);
		
		for (int i = 0; i < WARMUPITER; ++i)
		{
			fprintf(stderr, "\rFWARM %d", i);
			clEnqueueNDRangeKernel(env.queue(), kernf.kernel(), 2, NULL, ndims, NULL, 0, NULL, &rune);
			clWaitForEvents(1, &rune);
		}
		clEnqueueReadImage(env.txQueue(), img1.block(), CL_TRUE, origin, endpoint, 0, 0, ihost1, 0, NULL, &ree);
		clWaitForEvents(1, &ree);

		cerr << "\nfloat kernel:" << endl;

		stime = _x_time();
		clEnqueueWriteImage(env.txQueue(), img0.block(), CL_TRUE, origin, endpoint, 0, 0, ihost0, 0, NULL, &wre);

		for (int i = 0; i < ITER; ++i)
		{
			fprintf(stderr, "\rITER %d", i);
			clEnqueueNDRangeKernel(env.queue(), kernf.kernel(), 2, NULL, ndims, NULL, 0, NULL, &rune);
			clWaitForEvents(1, &rune);

			CLEV_TIME(rune, runtime);
		}
		clEnqueueReadImage(env.txQueue(), img1.block(), CL_TRUE, origin, endpoint, 0, 0, ihost1, 0, NULL, &ree);
		clWaitForEvents(1, &ree);

		etime = _x_time();
		fprintf(stderr, "\n");

		cerr << "RUN: " << runtime / ITER << endl;
		cerr << "CTM: " << (double)(etime-stime)/1000000ll / ITER << endl;
		cerr << "FPS: " << ITER * 1000 / runtime << endl;

		
		ih0.unmap();
		ih1.unmap();

	} catch (string s) {
_X_TIMER_TEARDOWN
		cerr << s << endl;
		return 1;
	}

_X_TIMER_TEARDOWN

	return 0;
}
