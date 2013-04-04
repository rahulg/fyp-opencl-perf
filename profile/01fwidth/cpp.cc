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
	cl_ushort filter[FWIDTH];
	
	cl_uchar *host0 = static_cast<cl_uchar*>(valloc(sizeof(cl_uchar) * (WIDTH+FWIDTH) * HEIGHT));
	cl_uchar *host1 = static_cast<cl_uchar*>(valloc(sizeof(cl_uchar) * (WIDTH+FWIDTH) * HEIGHT));

	for (int i = 0; i < (WIDTH+FWIDTH)*HEIGHT; ++i)
	{
		host0[i] = i % 256;
	}

	for (int i = 0; i < FWIDTH; ++i)
	{
		filter[i] = 4;
	}

	double wrtime=0.0, runtime=0.0, retime=0.0;
	uint64_t stime, etime;

	for (int i = 0; i < WARMUPITER; ++i)
	{
		fprintf(stderr, "\rWARM %d", i);
		for (int y = 0; y < HEIGHT; ++y)
		{
			for (int x = 0; x < WIDTH; ++x)
			{
				int val = 0;
				for (int k = 0; k < FWIDTH; ++k)
				{
					val += host0[y*WIDTH + (x+k)] * filter[k];
				}
				host1[y*WIDTH + x] = val;
			}
		}
	}

	stime = _x_time();

	for (int i = 0; i < ITER; ++i)
	{
		fprintf(stderr, "\rITER %d", i);
		for (int y = 0; y < HEIGHT; ++y)
		{
			for (int x = 0; x < WIDTH; ++x)
			{
				int val = 0;
				for (int k = 0; k < FWIDTH; ++k)
				{
					val += host0[y*WIDTH + (x+k)] * filter[k];
				}
				host1[y*WIDTH + x] = val;
			}
		}
	}

	etime = _x_time();
	fprintf(stderr, "\n");

	uint64_t sum;
	for (int i = 0; i < WIDTH * HEIGHT; ++i)
	{
		sum += host1[i];
	}

	cerr << sum << endl;

	cerr << "[c++] CTM: " << (double)(etime-stime)/1000000ll / ITER << endl;
	cout << "[c++] WID: " << FWIDTH << endl;
	cout << "[c++] FPS: " << ITER * 1000 / ((double)(etime-stime)/1000000ll) << endl;

_X_TIMER_TEARDOWN

	return 0;
}
