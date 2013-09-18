/*
	OpenCL Utilities
*/

#ifndef OCL_UTILS_H
#define OCL_UTILS_H

#include <stdio.h>
#include <CL/opencl.h>

#define OCL_CHECK(x) oclcheck((x),__FILE__,__LINE__)


#ifdef __cplusplus
extern "C" {
#endif

void oclcheck(int err, const char *file, const int linenum);





#ifdef __cplusplus
}
#endif

#endif




