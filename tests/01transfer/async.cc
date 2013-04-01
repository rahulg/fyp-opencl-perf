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

	size_t bufsz = WIDTH * HEIGHT * CHANNELS;
	size_t iter = ITER;

	
	try {

		Environment env(DeviceType::GPU);
		Program pg(env, "testpgm.cl");
		Kernel krn(pg, "copy");
		Kernel krn2(pg, "copy");
		Kernel ikrn(pg, "intense");
		Kernel ikrn2(pg, "intense");

		Buffer<cl_float> data0(env, MemoryType::ReadOnly, bufsz);
		Buffer<cl_float> data1(env, MemoryType::WriteOnly, bufsz);
		Buffer<cl_float> data2(env, MemoryType::ReadOnly, bufsz);
		Buffer<cl_float> data3(env, MemoryType::WriteOnly, bufsz);

		krn.setArgumentBuffer(0, data0);
		krn.setArgumentBuffer(1, data1);
		krn2.setArgumentBuffer(0, data2);
		krn2.setArgumentBuffer(1, data3);
		ikrn.setArgumentBuffer(0, data0);
		ikrn.setArgumentBuffer(1, data1);
		ikrn2.setArgumentBuffer(0, data2);
		ikrn2.setArgumentBuffer(1, data3);

		Buffer<cl_float> h0(env, MemoryType::PinnedReadWrite, bufsz);
		Buffer<cl_float> h1(env, MemoryType::PinnedReadWrite, bufsz);
		h0.map(MapMode::Write);
		h1.map(MapMode::Read);
		cl_float *host0 = h0.data();
		cl_float *host1 = h1.data();
		Buffer<cl_float> h2(env, MemoryType::PinnedReadWrite, bufsz);
		Buffer<cl_float> h3(env, MemoryType::PinnedReadWrite, bufsz);
		h2.map(MapMode::Write);
		h3.map(MapMode::Read);
		cl_float *host2 = h2.data();
		cl_float *host3 = h3.data();

		for (int i = 0; i < bufsz; ++i)
		{
			host0[i] = i;
			host2[i] = i;
		}

		cl_ulong start, end;
		double wrtime=0.0, runtime=0.0, retime=0.0;
		uint64_t stime, etime;
		cl_event ev[6] = {NULL, NULL, NULL, NULL, NULL, NULL};

		for (int i = 0; i < WARMUPITER/2; ++i)
		{
			fprintf(stderr, "\rWARM %d", i*2);
			clEnqueueWriteBuffer(env.txQueue(), data0.block(), CL_FALSE, 0, data0.size(), host0, 0, NULL, &ev[0]);
			clEnqueueNDRangeKernel(env.queue(), krn.kernel(), 1, NULL, &iter, NULL, 1, &ev[0], &ev[1]);
			clEnqueueReadBuffer(env.txQueue(), data1.block(), CL_FALSE, 0, data1.size(), host1, 1, &ev[1], &ev[2]);

			clWaitForEvents(3, &ev[3]);
			if (i != 0) {
				CLEV_TIME(ev[3], wrtime);
				CLEV_TIME(ev[4], runtime);
				CLEV_TIME(ev[5], retime);
			}

			fprintf(stderr, "\rWARM %d", i*2+1);
			clEnqueueWriteBuffer(env.txQueue(), data2.block(), CL_FALSE, 0, data2.size(), host2, 0, NULL, &ev[3]);
			clEnqueueNDRangeKernel(env.queue(), krn2.kernel(), 1, NULL, &iter, NULL, 1, &ev[3], &ev[4]);
			clEnqueueReadBuffer(env.txQueue(), data3.block(), CL_FALSE, 0, data3.size(), host3, 1, &ev[4], &ev[5]);

			clWaitForEvents(3, &ev[0]);
			CLEV_TIME(ev[0], wrtime);
			CLEV_TIME(ev[1], runtime);
			CLEV_TIME(ev[2], retime);
			
		}

		clWaitForEvents(1, &ev[3]);

		cerr << "\nCopy kernel:" << endl;

		stime = _x_time();

		for (int i = 0; i < ITER/2; ++i)
		{
			fprintf(stderr, "\rITER %d", i*2);
			clEnqueueWriteBuffer(env.txQueue(), data0.block(), CL_FALSE, 0, data0.size(), host0, 0, NULL, &ev[0]);
			clEnqueueNDRangeKernel(env.queue(), krn.kernel(), 1, NULL, &iter, NULL, 1, &ev[0], &ev[1]);
			clEnqueueReadBuffer(env.txQueue(), data1.block(), CL_FALSE, 0, data1.size(), host1, 1, &ev[1], &ev[2]);

			clWaitForEvents(3, &ev[3]);
			if (i != 0) {
				CLEV_TIME(ev[3], wrtime);
				CLEV_TIME(ev[4], runtime);
				CLEV_TIME(ev[5], retime);
			}

			fprintf(stderr, "\rITER %d", i*2+1);
			clEnqueueWriteBuffer(env.txQueue(), data2.block(), CL_FALSE, 0, data2.size(), host2, 0, NULL, &ev[3]);
			clEnqueueNDRangeKernel(env.queue(), krn2.kernel(), 1, NULL, &iter, NULL, 1, &ev[3], &ev[4]);
			clEnqueueReadBuffer(env.txQueue(), data3.block(), CL_FALSE, 0, data3.size(), host3, 1, &ev[4], &ev[5]);

			clWaitForEvents(3, &ev[0]);
			CLEV_TIME(ev[0], wrtime);
			CLEV_TIME(ev[1], runtime);
			CLEV_TIME(ev[2], retime);
			
		}

		clWaitForEvents(1, &ev[3]);
		CLEV_TIME(ev[3], wrtime);
		clWaitForEvents(1, &ev[4]);
		CLEV_TIME(ev[4], runtime);
		clWaitForEvents(1, &ev[5]);
		CLEV_TIME(ev[5], retime);

		etime = _x_time();
		fprintf(stderr, "\n");

		cerr << "WRI: " << wrtime / ITER << endl;
		cerr << "RUN: " << runtime / ITER << endl;
		cerr << "REA: " << retime / ITER << endl;
		cerr << "CTM: " << (double)(etime-stime)/1000000ll / ITER << endl;
		cerr << "FPS: " << ITER * 1000 / ((double)(etime-stime)/1000000ll) << endl;

		cerr << "Intense kernel:" << endl;

		stime = _x_time();

		for (int i = 0; i < ITER/2; ++i)
		{
			fprintf(stderr, "\rITER %d", i*2);
			clEnqueueWriteBuffer(env.txQueue(), data0.block(), CL_FALSE, 0, data0.size(), host0, 0, NULL, &ev[0]);
			clEnqueueNDRangeKernel(env.queue(), ikrn.kernel(), 1, NULL, &iter, NULL, 1, &ev[0], &ev[1]);
			clEnqueueReadBuffer(env.txQueue(), data1.block(), CL_FALSE, 0, data1.size(), host1, 1, &ev[1], &ev[2]);

			clWaitForEvents(3, &ev[3]);
			if (i != 0) {
				CLEV_TIME(ev[3], wrtime);
				CLEV_TIME(ev[4], runtime);
				CLEV_TIME(ev[5], retime);
			}

			fprintf(stderr, "\rITER %d", i*2+1);
			clEnqueueWriteBuffer(env.txQueue(), data2.block(), CL_FALSE, 0, data2.size(), host2, 0, NULL, &ev[3]);
			clEnqueueNDRangeKernel(env.queue(), ikrn2.kernel(), 1, NULL, &iter, NULL, 1, &ev[3], &ev[4]);
			clEnqueueReadBuffer(env.txQueue(), data3.block(), CL_FALSE, 0, data3.size(), host3, 1, &ev[4], &ev[5]);

			clWaitForEvents(3, &ev[0]);
			CLEV_TIME(ev[0], wrtime);
			CLEV_TIME(ev[1], runtime);
			CLEV_TIME(ev[2], retime);
			
		}

		clWaitForEvents(1, &ev[3]);
		CLEV_TIME(ev[3], wrtime);
		clWaitForEvents(1, &ev[4]);
		CLEV_TIME(ev[4], runtime);
		clWaitForEvents(1, &ev[5]);
		CLEV_TIME(ev[5], retime);

		etime = _x_time();
		fprintf(stderr, "\n");

		cerr << "WRI: " << wrtime / ITER << endl;
		cerr << "RUN: " << runtime / ITER << endl;
		cerr << "REA: " << retime / ITER << endl;
		cerr << "CTM: " << (double)(etime-stime)/1000000ll / ITER << endl;
		cerr << "FPS: " << ITER * 1000 / ((double)(etime-stime)/1000000ll) << endl;


		h0.unmap();
		h1.unmap();
		h2.unmap();
		h3.unmap();

	} catch (string s) {
_X_TIMER_TEARDOWN
		cerr << s << endl;
		return 1;
	}

_X_TIMER_TEARDOWN

	return 0;
}
