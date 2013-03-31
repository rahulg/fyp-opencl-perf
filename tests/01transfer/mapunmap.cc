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
		Kernel krn(pg, "testkern");

		Buffer<cl_float> data0(env, MemoryType::ReadWrite, bufsz);
		Buffer<cl_float> data1(env, MemoryType::ReadWrite, bufsz);

		krn.setArgumentBuffer(0, data0);
		krn.setArgumentBuffer(1, data1);

		cl_float *host0 = new cl_float[bufsz];
		cl_float *host1 = new cl_float[bufsz];
		cl_float *dev0, *dev1;

		for (int i = 0; i < bufsz; ++i)
		{
			host0[i] = i;
		}

		cl_ulong start, end;
		double map0=0.0, unmap0=0.0, runtime=0.0, map1=0.0, unmap1=0.0;
		uint64_t stime, etime;
		cl_int err;

		stime = _x_time();

		for (int i = 0; i < ITER; ++i)
		{
			fprintf(stderr, "\rITER %d", i);
			cl_event mape0, unmape0, rune, mape1, unmape1;
			dev0 = static_cast<cl_float*>(clEnqueueMapBuffer(env.queue(), data0.block(), CL_TRUE, CL_MAP_WRITE, 0, data0.size(), 0, NULL, &mape0, &err));
			for (int i = 0; i < bufsz; ++i) {
				dev0[i] = host0[i];
			}
			clEnqueueUnmapMemObject(env.queue(), data0.block(), dev0, 0, NULL, &unmape0);
			clWaitForEvents(1, &unmape0);
			clEnqueueNDRangeKernel(env.queue(), krn.kernel(), 1, NULL, &iter, NULL, 0, NULL, &rune);
			clWaitForEvents(1, &rune);
			dev1 = static_cast<cl_float*>(clEnqueueMapBuffer(env.queue(), data1.block(), CL_TRUE, CL_MAP_READ, 0, data1.size(), 0, NULL, &mape1, &err));
			for (int i = 0; i < bufsz; ++i) {
				host1[i] = dev1[i];
			}
			clEnqueueUnmapMemObject(env.queue(), data1.block(), dev1, 0, NULL, &unmape1);
			clWaitForEvents(1, &unmape1);

			CLEV_TIME(mape0, map0);
			CLEV_TIME(unmape0, unmap0);
			CLEV_TIME(rune, runtime);
			CLEV_TIME(mape1, map1);
			CLEV_TIME(unmape1, unmap1);
		}

		etime = _x_time();
		fprintf(stderr, "\n");

		cerr << "MP0: " << map0 / ITER << endl;
		cerr << "UM0: " << unmap0 / ITER << endl;
		cerr << "RUN: " << runtime / ITER << endl;
		cerr << "MP1: " << map1 / ITER << endl;
		cerr << "UM1: " << unmap1 / ITER << endl;
		cerr << "CTM: " << (double)(etime-stime)/1000000ll / ITER << endl;
		cerr << "FPS: " << ITER * 1000 / ((double)(etime-stime)/1000000ll) << endl;

		delete[] host0;
		delete[] host1;

	} catch (string s) {
_X_TIMER_TEARDOWN
		cerr << s << endl;
		return 1;
	}

_X_TIMER_TEARDOWN

	return 0;
}
