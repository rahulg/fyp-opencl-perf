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
#define FWIDTH 12
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
	cl_ushort filt[8];
	cl_ushort4 filt4[2];
	
	try {

		sprintf(args, "-DFWIDTH=%d", FWIDTH);

		Environment env(DeviceType::GPU);
		Program pg(env, "testpgm.cl", args);
		Kernel krn(pg, "convolve");
		Kernel krn4(pg, "convolve4");

		Buffer<cl_ushort> filter(env, MemoryType::ReadOnly, FWIDTH);
		Buffer<cl_ushort4> filter0(env, MemoryType::ReadOnly, FWIDTH/4);
		Buffer<cl_ushort4> filter1(env, MemoryType::ReadOnly, FWIDTH/4);
		Buffer<cl_ushort4> filter2(env, MemoryType::ReadOnly, FWIDTH/4);
		Buffer<cl_ushort4> filter3(env, MemoryType::ReadOnly, FWIDTH/4);

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
		
		Image qimg0(env, MemoryType::ReadOnly, Channels::RGBA, PixelFormat::Unsigned8, {WIDTH/4, HEIGHT});
		Image qimg1(env, MemoryType::WriteOnly, Channels::RGBA, PixelFormat::Unsigned8, {WIDTH/4, HEIGHT});

		krn4.setArgumentBuffer(0, filter0);
		krn4.setArgumentBuffer(1, filter1);
		krn4.setArgumentBuffer(2, filter2);
		krn4.setArgumentBuffer(3, filter3);
		krn4.setArgumentImage(4, qimg0);
		krn4.setArgumentImage(5, qimg1);

		Image qh0(env, MemoryType::PinnedReadWrite, Channels::RGBA, PixelFormat::Unsigned8, {WIDTH/4, HEIGHT});
		Image qh1(env, MemoryType::PinnedReadWrite, Channels::RGBA, PixelFormat::Unsigned8, {WIDTH/4, HEIGHT});
		qh0.map(MapMode::Write);
		qh1.map(MapMode::Read);
		cl_uchar4 *qhost0 = static_cast<cl_uchar4*>(qh0.data());
		cl_uchar4 *qhost1 = static_cast<cl_uchar4*>(qh1.data());

		for (int i = 0; i < WIDTH*HEIGHT; ++i)
		{
			host0[i] = i % 256;
		}

		for (int i = 0; i < WIDTH*HEIGHT/4; ++i)
		{
			qhost0[i].s[0] = i % 256;
			qhost0[i].s[1] = i % 256;
			qhost0[i].s[2] = i % 256;
			qhost0[i].s[3] = i % 256;
		}

		for (int i = 0; i < FWIDTH; ++i)
		{
			filt[i] = 1;
		}

		for (int i = 0; i < FWIDTH/4; ++i)
		{
			filt4[i].s[0] = 1;
			filt4[i].s[1] = 1;
			filt4[i].s[2] = 1;
			filt4[i].s[3] = 1;
		}

		cl_event bufev;
		bufev = filter.queueWrite(filt);
		clWaitForEvents(1, &bufev);

		bufev = filter0.queueWrite(filt4);
		clWaitForEvents(1, &bufev);
		bufev = filter1.queueWrite(filt4);
		clWaitForEvents(1, &bufev);
		bufev = filter2.queueWrite(filt4);
		clWaitForEvents(1, &bufev);
		bufev = filter3.queueWrite(filt4);
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

		cerr << "[-vc] WRI: " << wrtime / ITER << endl;
		cerr << "[-vc] RUN: " << runtime / ITER << endl;
		cerr << "[-vc] REA: " << retime / ITER << endl;
		cerr << "[-vc] CTM: " << (double)(etime-stime)/1000000ll / ITER << endl;
		cout << "[-vc] WID: " << FWIDTH << endl;
		cout << "[-vc] FPS: " << ITER * 1000 / ((double)(etime-stime)/1000000ll) << endl;


		dim[0] = WIDTH/4;
		endpoint[0] = WIDTH/4;

		for (int i = 0; i < WARMUPITER; ++i)
		{
			fprintf(stderr, "\rWARM %d", i);
			cl_event wre, rune, ree;
			clEnqueueWriteImage(env.txQueue(), qimg0.block(), CL_TRUE, origin, endpoint, 0, 0, qhost0, 0, NULL, &wre);
			clEnqueueNDRangeKernel(env.queue(), krn4.kernel(), 2, NULL, dim, NULL, 0, NULL, &rune);
			clWaitForEvents(1, &rune);
			clEnqueueReadImage(env.txQueue(), qimg1.block(), CL_TRUE, origin, endpoint, 0, 0, qhost1, 0, NULL, &ree);
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

		cerr << "[+vc] WRI: " << wrtime / ITER << endl;
		cerr << "[+vc] RUN: " << runtime / ITER << endl;
		cerr << "[+vc] REA: " << retime / ITER << endl;
		cerr << "[+vc] CTM: " << (double)(etime-stime)/1000000ll / ITER << endl;
		cout << "[+vc] WID: " << FWIDTH << endl;
		cout << "[+vc] FPS: " << ITER * 1000 / ((double)(etime-stime)/1000000ll) << endl;

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
