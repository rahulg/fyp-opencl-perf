#ifndef __TIMING_H__
#define __TIMING_H__

#if defined(__APPLE__)
	#include <mach/clock.h>
	#include <mach/mach.h>
	clock_serv_t cclock;
	mach_timespec_t tspec;
#elif defined(__linux__)
	struct timespec tspec;
#else
	struct timeval tv;
#endif

uint64_t _x_time()
{
#if defined(__linux__)
	clock_gettime(CLOCK_MONOTONIC, &tspec);
#elif defined(__APPLE__)
	clock_get_time(cclock, &tspec);
#else
	#warning "Not running OS X or Linux. Timer @ usec precision."
	gettimeofday(&tv, NULL);
#endif

#if defined(__linux__) || defined(__APPLE__)
	return (uint64_t)(tspec.tv_nsec + (uint64_t)tspec.tv_sec * 1000000000ll);
#else
	return (uint64_t)(tv.tv_usec * 1000 + (uint64_t)tv.tv_sec * 1000000000ll);
#endif
}

#ifdef __APPLE__

	#define _X_TIMER_SETUP host_get_clock_service(mach_host_self(), SYSTEM_CLOCK, &cclock);
	#define _X_TIMER_TEARDOWN mach_port_deallocate(mach_task_self(), cclock);

#else
	
	#define _X_TIMER_SETUP 
	#define _X_TIMER_TEARDOWN 

#endif

#endif
