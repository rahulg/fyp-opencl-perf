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

#ifndef FWIDTH
#define FWIDTH 6
#endif

#define CLEV_TIME(ev,dbl) \
clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL); \
clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL); \
dbl += (double)(end - start)*(double)(1e-06);

int main(int argc, char const *argv[])
{

_X_TIMER_SETUP

	size_t iter = ITER;
	char args[255];
	cl_ushort filt[FWIDTH];
	
	try {

		sprintf(args, "-DFWIDTH=%d", FWIDTH);

		Environment env(DeviceType::GPU);
		Program pg(env, "testpgm.cl", args);
		Kernel krn(pg, "convolve");

		Buffer<cl_ushort> filter(env, MemoryType::ReadOnly, FWIDTH);

		Image img0(env, MemoryType::ReadOnly, Channels::Single, PixelFormat::Unsigned8, {WIDTH, HEIGHT});
		Image img1(env, MemoryType::WriteOnly, Channels::Single, PixelFormat::Unsigned8, {WIDTH, HEIGHT});

		krn.setArgumentBuffer(0, filter);
		krn.setArgumentImage(1, img0);
		krn.setArgumentImage(2, img1);

		Image h0(env, MemoryType::PinnedReadWrite, Channels::Single, PixelFormat::Unsigned8, {WIDTH, HEIGHT});
		Image h1(env, MemoryType::PinnedReadWrite, Channels::Single, PixelFormat::Unsigned8, {WIDTH, HEIGHT});
		h0.map(MapMode::Write);
		h1.map(MapMode::Read);
		cl_uchar *host0 = static_cast<cl_uchar*>(h0.data());
		cl_uchar *host1 = static_cast<cl_uchar*>(h1.data());

		for (int i = 0; i < WIDTH*HEIGHT; ++i)
		{
			host0[i] = i % 256;
		}

		for (int i = 0; i < FWIDTH; ++i)
		{
			filt[i] = 1;
		}

		cl_event bufev;
		bufev = filter.queueWrite(filt);
		clWaitForEvents(1, &bufev);

		cl_ulong start, end;
		double wrtime=0.0, runtime=0.0, retime=0.0;
		uint64_t stime, etime;
		size_t dim[2] = {WIDTH, HEIGHT};
		size_t origin[3] = {0,0,0};
		size_t endpoint[3] = {WIDTH,HEIGHT,1};

		for (int i = 0; i < WARMUPITER; ++i)
		{
			fprintf(stderr, "\rWARM %d", i);
			cl_event wre, rune, ree;
			clEnqueueWriteImage(env.txQueue(), img0.block(), CL_TRUE, origin, endpoint, 0, 0, host0, 0, NULL, &wre);
			clEnqueueNDRangeKernel(env.queue(), krn.kernel(), 2, NULL, dim, NULL, 0, NULL, &rune);
			clWaitForEvents(1, &rune);
			clEnqueueReadImage(env.txQueue(), img1.block(), CL_TRUE, origin, endpoint, 0, 0, host1, 0, NULL, &ree);
			clWaitForEvents(1, &ree);

			CLEV_TIME(wre, wrtime);
			CLEV_TIME(rune, runtime);
			CLEV_TIME(ree, retime);
		}

		stime = _x_time();

		for (int i = 0; i < ITER; ++i)
		{
			fprintf(stderr, "\rITER %d", i);
			cl_event wre, rune, ree;
			clEnqueueWriteImage(env.txQueue(), img0.block(), CL_TRUE, origin, endpoint, 0, 0, host0, 0, NULL, &wre);
			clEnqueueNDRangeKernel(env.queue(), krn.kernel(), 2, NULL, dim, NULL, 0, NULL, &rune);
			clWaitForEvents(1, &rune);
			clEnqueueReadImage(env.txQueue(), img1.block(), CL_TRUE, origin, endpoint, 0, 0, host1, 0, NULL, &ree);
			clWaitForEvents(1, &ree);

			CLEV_TIME(wre, wrtime);
			CLEV_TIME(rune, runtime);
			CLEV_TIME(ree, retime);
		}

		etime = _x_time();
		fprintf(stderr, "\n");

		cerr << "[ocl] WRI: " << wrtime / ITER << endl;
		cerr << "[ocl] RUN: " << runtime / ITER << endl;
		cerr << "[ocl] REA: " << retime / ITER << endl;
		cerr << "[ocl] CTM: " << (double)(etime-stime)/1000000ll / ITER << endl;
		cout << "[ocl] DIM: " << WIDTH << "x" << HEIGHT << endl;
		cout << "[ocl] FPS: " << ITER * 1000 / ((double)(etime-stime)/1000000ll) << endl;

		h0.unmap();
		h1.unmap();

	} catch (string s) {
_X_TIMER_TEARDOWN
		cerr << s << endl;
		return 1;
	}

_X_TIMER_TEARDOWN

	return 0;
}
