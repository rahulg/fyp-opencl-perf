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

		Environment env(DeviceType::CPU);
		Program pg(env, "testpgm.cl");
		Kernel krn(pg, "copy");
		Kernel ikrn(pg, "intense");

		Buffer<cl_float> data0(env, MemoryType::ReadOnly, bufsz);
		Buffer<cl_float> data1(env, MemoryType::WriteOnly, bufsz);

		krn.setArgumentBuffer(0, data0);
		krn.setArgumentBuffer(1, data1);
		ikrn.setArgumentBuffer(0, data0);
		ikrn.setArgumentBuffer(1, data1);

		cl_float *host0 = (cl_float*)valloc(sizeof(cl_float) * bufsz);
		cl_float *host1 = (cl_float*)valloc(sizeof(cl_float) * bufsz);

		for (int i = 0; i < bufsz; ++i)
		{
			host0[i] = i;
		}

		cl_ulong start, end;
		double wrtime=0.0, runtime=0.0, retime=0.0;
		uint64_t stime, etime;

		for (int i = 0; i < WARMUPITER; ++i)
		{
			fprintf(stderr, "\rWARM %d", i);
			cl_event wre, rune, ree;
			clEnqueueWriteBuffer(env.txQueue(), data0.block(), CL_TRUE, 0, data0.size(), host0, 0, NULL, &wre);
			clEnqueueNDRangeKernel(env.queue(), krn.kernel(), 1, NULL, &iter, NULL, 0, NULL, &rune);
			clWaitForEvents(1, &rune);
			clEnqueueReadBuffer(env.txQueue(), data1.block(), CL_TRUE, 0, data1.size(), host1, 0, NULL, &ree);
			clWaitForEvents(1, &ree);

			CLEV_TIME(wre, wrtime);
			CLEV_TIME(rune, runtime);
			CLEV_TIME(ree, retime);
		}

		cerr << "\nCopy kernel:" << endl;

		stime = _x_time();

		for (int i = 0; i < ITER; ++i)
		{
			fprintf(stderr, "\rITER %d", i);
			cl_event wre, rune, ree;
			clEnqueueWriteBuffer(env.txQueue(), data0.block(), CL_TRUE, 0, data0.size(), host0, 0, NULL, &wre);
			clEnqueueNDRangeKernel(env.queue(), krn.kernel(), 1, NULL, &iter, NULL, 0, NULL, &rune);
			clWaitForEvents(1, &rune);
			clEnqueueReadBuffer(env.txQueue(), data1.block(), CL_TRUE, 0, data1.size(), host1, 0, NULL, &ree);
			clWaitForEvents(1, &ree);

			CLEV_TIME(wre, wrtime);
			CLEV_TIME(rune, runtime);
			CLEV_TIME(ree, retime);
		}

		etime = _x_time();
		fprintf(stderr, "\n");

		cerr << "WRI: " << wrtime / ITER << endl;
		cerr << "RUN: " << runtime / ITER << endl;
		cerr << "REA: " << retime / ITER << endl;
		cerr << "CTM: " << (double)(etime-stime)/1000000ll / ITER << endl;
		cerr << "FPS: " << ITER * 1000 / ((double)(etime-stime)/1000000ll) << endl;

		cerr << "Intense kernel:" << endl;

		stime = _x_time();

		for (int i = 0; i < ITER; ++i)
		{
			fprintf(stderr, "\rITER %d", i);
			cl_event wre, rune, ree;
			clEnqueueWriteBuffer(env.txQueue(), data0.block(), CL_TRUE, 0, data0.size(), host0, 0, NULL, &wre);
			clEnqueueNDRangeKernel(env.queue(), ikrn.kernel(), 1, NULL, &iter, NULL, 0, NULL, &rune);
			clWaitForEvents(1, &rune);
			clEnqueueReadBuffer(env.txQueue(), data1.block(), CL_TRUE, 0, data1.size(), host1, 0, NULL, &ree);
			clWaitForEvents(1, &ree);

			CLEV_TIME(wre, wrtime);
			CLEV_TIME(rune, runtime);
			CLEV_TIME(ree, retime);
		}

		etime = _x_time();
		fprintf(stderr, "\n");

		cerr << "WRI: " << wrtime / ITER << endl;
		cerr << "RUN: " << runtime / ITER << endl;
		cerr << "REA: " << retime / ITER << endl;
		cerr << "CTM: " << (double)(etime-stime)/1000000ll / ITER << endl;
		cerr << "FPS: " << ITER * 1000 / ((double)(etime-stime)/1000000ll) << endl;

		free(host0);
		free(host1);

	} catch (string s) {
_X_TIMER_TEARDOWN
		cerr << s << endl;
		return 1;
	}

_X_TIMER_TEARDOWN

	return 0;
}
