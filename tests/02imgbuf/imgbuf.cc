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

		// BUFFER
		Kernel kernb(pg, "compb");
		Buffer<cl_uchar4> data0(env, MemoryType::ReadOnly, bufsz*2);
		Buffer<cl_uchar4> data1(env, MemoryType::WriteOnly, bufsz*2);

		kernb.setArgumentBuffer(0, data0);
		kernb.setArgumentBuffer(1, data1);

		Buffer<cl_uchar4> h0(env, MemoryType::PinnedReadWrite, bufsz*2);
		Buffer<cl_uchar4> h1(env, MemoryType::PinnedReadWrite, bufsz*2);
		h0.map(MapMode::Write);
		h1.map(MapMode::Read);
		cl_uchar4 *host0 = h0.data();
		cl_uchar4 *host1 = h1.data();

		// IMAGE
		Kernel kernp(pg, "compp");
		Image img0(env, MemoryType::ReadOnly, Channels::RGBA, PixelFormat::Unsigned8, {WIDTH, HEIGHT*2});
		Image img1(env, MemoryType::WriteOnly, Channels::RGBA, PixelFormat::Unsigned8, {WIDTH, HEIGHT*2});

		kernp.setArgumentImage(0, img0);
		kernp.setArgumentImage(1, img1);

		Image ih0(env, MemoryType::PinnedReadWrite, Channels::RGBA, PixelFormat::Unsigned8, {WIDTH, HEIGHT*2});
		Image ih1(env, MemoryType::PinnedReadWrite, Channels::RGBA, PixelFormat::Unsigned8, {WIDTH, HEIGHT*2});
		ih0.map(MapMode::Write);
		ih1.map(MapMode::Read);
		cl_uchar *ihost0 = static_cast<cl_uchar*>(ih0.data());
		cl_uchar *ihost1 = static_cast<cl_uchar*>(ih1.data());

		for (int i = 0; i < bufsz*2; ++i) {
			host0[i].s[0] = i % 256;
			host0[i].s[1] = i % 256;
			host0[i].s[2] = i % 256;
			host0[i].s[3] = i % 256;
		}
		for (int i = 0; i < bufsz * 8; ++i) {
			ihost0[i] = i % 256;
		}

		cl_ulong start, end;
		double wrtime=0.0, runtime=0.0, retime=0.0;
		uint64_t stime, etime;
		size_t ndims[2] = {WIDTH,HEIGHT};

		/****************IMAGE******************/
		size_t origin[3] = {0,0,0};
		size_t endpoint[3] = {WIDTH,HEIGHT,1};
		cl_event wre, rune, ree;
		clEnqueueWriteImage(env.txQueue(), img0.block(), CL_TRUE, origin, endpoint, 0, 0, ihost0, 0, NULL, &wre);
		

		for (int i = 0; i < WARMUPITER; ++i)
		{
			fprintf(stderr, "\rIWARM %d", i);
			clEnqueueNDRangeKernel(env.queue(), kernp.kernel(), 2, NULL, ndims, NULL, 0, NULL, &rune);
			clWaitForEvents(1, &rune);
		}
		clEnqueueReadImage(env.txQueue(), img1.block(), CL_TRUE, origin, endpoint, 0, 0, ihost1, 0, NULL, &ree);
		clWaitForEvents(1, &ree);

		cerr << "\nIIntense kernel:" << endl;

		stime = _x_time();
		clEnqueueWriteImage(env.txQueue(), img0.block(), CL_TRUE, origin, endpoint, 0, 0, ihost0, 0, NULL, &wre);

		for (int i = 0; i < ITER; ++i)
		{
			fprintf(stderr, "\rITER %d", i);
			clEnqueueNDRangeKernel(env.queue(), kernp.kernel(), 2, NULL, ndims, NULL, 0, NULL, &rune);
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

		/*************BUFFER*************/
		clEnqueueWriteBuffer(env.txQueue(), data0.block(), CL_TRUE, 0, data0.size(), host0, 0, NULL, &wre);

		for (int i = 0; i < WARMUPITER; ++i)
		{
			fprintf(stderr, "\rBWARM %d", i);
			clEnqueueNDRangeKernel(env.queue(), kernb.kernel(), 2, NULL, ndims, NULL, 0, NULL, &rune);
			clWaitForEvents(1, &rune);
		}

		clEnqueueReadBuffer(env.txQueue(), data1.block(), CL_TRUE, 0, data1.size(), host1, 0, NULL, &ree);
		clWaitForEvents(1, &ree);

		cerr << "\nBIntense kernel:" << endl;

		stime = _x_time();
		clEnqueueWriteBuffer(env.txQueue(), data0.block(), CL_TRUE, 0, data0.size(), host0, 0, NULL, &wre);

		for (int i = 0; i < ITER; ++i)
		{
			fprintf(stderr, "\rITER %d", i);
			clEnqueueNDRangeKernel(env.queue(), kernb.kernel(), 2, NULL, ndims, NULL, 0, NULL, &rune);
			clWaitForEvents(1, &rune);

			CLEV_TIME(rune, runtime);
		}
		clEnqueueReadBuffer(env.txQueue(), data1.block(), CL_TRUE, 0, data1.size(), host1, 0, NULL, &ree);
		clWaitForEvents(1, &ree);

		etime = _x_time();
		fprintf(stderr, "\n");

		cerr << "RUN: " << runtime / ITER << endl;
		cerr << "CTM: " << (double)(etime-stime)/1000000ll / ITER << endl;
		cerr << "FPS: " << ITER * 1000 / runtime << endl;

		
		ih0.unmap();
		ih1.unmap();
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
