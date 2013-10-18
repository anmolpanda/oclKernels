#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include <CL/opencl.h>

#include "../../Utils/c_utils.h"
#include "../../Utils/ocl_utils.h"

#define TILE 64
#define TILE16 16 

//-----------------------------
//    A(NxT)  *  B (TxN) = C
//-----------------------------

void run1(int N); // one kernel
//void run2(int N); // two kernels
void run3(int N);
void run4(int N);


int main (int argc, char *argv[])
{
	if (argc !=  2){
		printf("Type ./Program N(square matrix width)\n");
		exit(1);
	}	

	int N = atoi(argv[1]); // row

	//run1(N);
	//
	
	//run2(N);
	//
	
	//run3(N);

	run4(N); // best performance




	return  0;
}

/*
void run2(int N)
{
	puts("lauching 1d threads");

	float *A;
	A = (float*)malloc(sizeof(float)*N*N);
	init_2d_f(A,N,N,1.f);

#ifdef DEBUG
	puts("A");
	check_2d_f(A,N,N);
#endif

	float *C;
	C = (float*)malloc(sizeof(float)*N*N);

	int blks = (N+63)/64;

	float *sum;
	sum = (float*)malloc(sizeof(float)*(blks+1));


	int NumK = 2;
	int NumE = 2;

	int i;

	double gpuTime;
	cl_ulong gstart, gend;

	//------------------------------------------------
	//  OpenCL 
	//------------------------------------------------
	cl_int err;

	cl_platform_id platform;          // OpenCL platform
	cl_device_id device_id;           // device ID
	cl_context context;               // context
	cl_command_queue queue;           // command queue
	cl_program program;               // program

	cl_kernel *kernel = (cl_kernel*)malloc(sizeof(cl_kernel)*NumK);

	cl_event *event = (cl_event*)malloc(sizeof(cl_event)*NumE);    

	// read kernel file
	char *fileName = "kernel.cl";
	char *kernelSource;
	size_t size;
	FILE *fh = fopen(fileName, "rb");
	if(!fh) {
		printf("Error: Failed to open kernel file!\n");
		exit(1);
	}
	fseek(fh,0,SEEK_END);
	size=ftell(fh);
	fseek(fh,0,SEEK_SET);
	kernelSource = malloc(size+1);
	size_t result;
	result = fread(kernelSource,1,size,fh);
	if(result != size){ fputs("Reading error", stderr);exit(1);}
	kernelSource[size] = '\0';
	
	// Bind to platform
	err = clGetPlatformIDs(1, &platform, NULL);
	OCL_CHECK(err);

	// Get ID for the device
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	OCL_CHECK(err);

	// Create a context  
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	OCL_CHECK(err);

	// Create a command queue 
	queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
	OCL_CHECK(err);

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &err);
	OCL_CHECK(err);

	// turn on optimization for kernel
	char *options="-cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros -cl-unsafe-math-optimizations -cl-finite-math-only";

	err = clBuildProgram(program, 1, &device_id, options, NULL, NULL);
	if(err != CL_SUCCESS)
		printCompilerOutput(program, device_id);
	OCL_CHECK(err);

	kernel[0] = clCreateKernel(program, "reduction_2a", &err);
	OCL_CHECK(err);

	kernel[1] = clCreateKernel(program, "reduction_2b", &err);
	OCL_CHECK(err);

	// memory on device
	cl_mem A_d   = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*N,  NULL, NULL);
	cl_mem sum_d = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*(blks+1),      NULL, NULL);

	// Initialize device memory
	err = clEnqueueWriteBuffer(queue, A_d, 	CL_TRUE, 0, sizeof(float)*N*N, 	A, 0, NULL, &event[0]); 
	OCL_CHECK(err);

	size_t localsize;
	size_t globalsize;

	localsize = 64;
	globalsize = ((N+63)/64)*64;

	err  = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &A_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &sum_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 2, sizeof(float)*64, NULL);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 3, sizeof(int), &N);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 4, sizeof(int), &blks);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	size_t k2_localsize;
	size_t k2_globalsize;

	k2_localsize = 64;
	k2_globalsize = ((blks+63)/64)*64;

	err  = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), &sum_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[1], 1, sizeof(float)*64, NULL);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[1], 2, sizeof(int), &N);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[1], 3, sizeof(int), &blks);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}


	err = clEnqueueNDRangeKernel(queue, kernel[0], 1, NULL, &globalsize, &localsize, 0, NULL, NULL);
	OCL_CHECK(err);

	err = clEnqueueNDRangeKernel(queue, kernel[1], 1, NULL, &k2_globalsize, &k2_localsize, 0, NULL, NULL);


	clFinish(queue);

	clEnqueueReadBuffer(queue, sum_d, CL_TRUE, 0, sizeof(float)*(blks+1), sum, 0, NULL , &event[1]);

	err = clWaitForEvents(1,&event[1]);
	OCL_CHECK(err);

	err = clGetEventProfilingInfo (event[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &gstart, NULL);
	OCL_CHECK(err);

	err = clGetEventProfilingInfo (event[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &gend, NULL);
	OCL_CHECK(err);

	gpuTime = (double)(gend -gstart)/1000000000.0;

	printf("oclTime = %lf (s)\n", gpuTime );


	//check_1d_f(sum, blks+1);

#ifdef DEBUG
	//puts("C");
	//check_2d_f(C,N,N);

#endif


	// free
	clReleaseMemObject(A_d);	
	clReleaseMemObject(sum_d);


	clReleaseProgram(program);
	clReleaseContext(context);
	clReleaseCommandQueue(queue);
	for(i=0;i<NumK;++i){
		clReleaseKernel(kernel[i]);
	}
	for(i=0;i<NumE;++i){
		clReleaseEvent(event[i]);
	}
	free(kernelSource);



	free(A);
	free(sum);

	return;
}
*/


void run1(int N)
{
	puts("run 1\n");


	float *A;
	A = (float*)malloc(sizeof(float)*N*N);
	init_2d_f(A,N,N,1.f);

#ifdef DEBUG
	puts("A");
	check_2d_f(A,N,N);
#endif

	//float *C;
	//C = (float*)malloc(sizeof(float)*N*N);

	int blks = (N+63)/64;

	float *sum;
	sum = (float*)malloc(sizeof(float)*(blks+1));

	int NumK = 1;
	int NumE = 2;

	int i;

	double gpuTime;
	cl_ulong gstart, gend;

	//------------------------------------------------
	//  OpenCL 
	//------------------------------------------------
	cl_int err;

	cl_platform_id platform;          // OpenCL platform
	cl_device_id device_id;           // device ID
	cl_context context;               // context
	cl_command_queue queue;           // command queue
	cl_program program;               // program

	cl_kernel *kernel = (cl_kernel*)malloc(sizeof(cl_kernel)*NumK);

	cl_event *event = (cl_event*)malloc(sizeof(cl_event)*NumE);    

	// read kernel file
	char *fileName = "kernel.cl";
	char *kernelSource;
	size_t size;
	FILE *fh = fopen(fileName, "rb");
	if(!fh) {
		printf("Error: Failed to open kernel file!\n");
		exit(1);
	}
	fseek(fh,0,SEEK_END);
	size=ftell(fh);
	fseek(fh,0,SEEK_SET);
	kernelSource = malloc(size+1);
	size_t result;
	result = fread(kernelSource,1,size,fh);
	if(result != size){ fputs("Reading error", stderr);exit(1);}
	kernelSource[size] = '\0';
	
	// Bind to platform
	err = clGetPlatformIDs(1, &platform, NULL);
	OCL_CHECK(err);

	// Get ID for the device
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	OCL_CHECK(err);

	// Create a context  
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	OCL_CHECK(err);

	// Create a command queue 
	queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
	OCL_CHECK(err);

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &err);
	OCL_CHECK(err);

	// turn on optimization for kernel
	char *options="-cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros -cl-unsafe-math-optimizations -cl-finite-math-only";

	err = clBuildProgram(program, 1, &device_id, options, NULL, NULL);
	if(err != CL_SUCCESS)
		printCompilerOutput(program, device_id);
	OCL_CHECK(err);

	kernel[0] = clCreateKernel(program, "reduction_1", &err);
	OCL_CHECK(err);


	// memory on device
	cl_mem A_d   = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*N,  NULL, NULL);
	cl_mem sum_d = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*(blks+1),      NULL, NULL);

	// Initialize device memory
	err = clEnqueueWriteBuffer(queue, A_d, 	CL_TRUE, 0, sizeof(float)*N*N, 	A, 0, NULL, &event[0]); 
	OCL_CHECK(err);

	size_t localsize;
	size_t globalsize;

	localsize = 64;
	globalsize = ((N+63)/64)*64;

	err  = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &A_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &sum_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 2, sizeof(float)*64, NULL);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 3, sizeof(int), &N);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 4, sizeof(int), &blks);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}


	err = clEnqueueNDRangeKernel(queue, kernel[0], 1, NULL, &globalsize, &localsize, 0, NULL, NULL);
	OCL_CHECK(err);

	clFinish(queue);

	clEnqueueReadBuffer(queue, sum_d, CL_TRUE, 0, sizeof(float)*(blks+1), sum, 0, NULL , &event[1]);

	err = clWaitForEvents(1,&event[1]);
	OCL_CHECK(err);

	err = clGetEventProfilingInfo (event[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &gstart, NULL);
	OCL_CHECK(err);

	err = clGetEventProfilingInfo (event[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &gend, NULL);
	OCL_CHECK(err);

	gpuTime = (double)(gend -gstart)/1000000000.0;

	printf("oclTime = %lf (s)\n", gpuTime );


	//check_1d_f(sum, blks+1);

#ifdef DEBUG
	//puts("C");
	//check_2d_f(C,N,N);

#endif


	// free
	clReleaseMemObject(A_d);	
	//clReleaseMemObject(C_d);
	clReleaseMemObject(sum_d);


	clReleaseProgram(program);
	clReleaseContext(context);
	clReleaseCommandQueue(queue);
	for(i=0;i<NumK;++i){
		clReleaseKernel(kernel[i]);
	}
	for(i=0;i<NumE;++i){
		clReleaseEvent(event[i]);
	}
	free(kernelSource);



	free(A);
	//free(C);
	free(sum);

	return;
}




void run3(int N)
{
	puts("run 3\n");

	float *A;
	A = (float*)malloc(sizeof(float)*N*N);
	init_2d_f(A,N,N,1.f);

#ifdef DEBUG
	puts("A");
	check_2d_f(A,N,N);
#endif

	int blks = (N+TILE-1)/TILE;

	float *sum;
	sum = (float*)malloc(sizeof(float));

	int NumK = 2;
	int NumE = 2;

	int i;

	double gpuTime;
	cl_ulong gstart, gend;

	//------------------------------------------------
	//  OpenCL 
	//------------------------------------------------
	cl_int err;

	cl_platform_id platform;          // OpenCL platform
	cl_device_id device_id;           // device ID
	cl_context context;               // context
	cl_command_queue queue;           // command queue
	cl_program program;               // program

	cl_kernel *kernel = (cl_kernel*)malloc(sizeof(cl_kernel)*NumK);

	cl_event *event = (cl_event*)malloc(sizeof(cl_event)*NumE);    

	// read kernel file
	char *fileName = "kernel.cl";
	char *kernelSource;
	size_t size;
	FILE *fh = fopen(fileName, "rb");
	if(!fh) {
		printf("Error: Failed to open kernel file!\n");
		exit(1);
	}
	fseek(fh,0,SEEK_END);
	size=ftell(fh);
	fseek(fh,0,SEEK_SET);
	kernelSource = malloc(size+1);
	size_t result;
	result = fread(kernelSource,1,size,fh);
	if(result != size){ fputs("Reading error", stderr);exit(1);}
	kernelSource[size] = '\0';
	
	// Bind to platform
	err = clGetPlatformIDs(1, &platform, NULL);
	OCL_CHECK(err);

	// Get ID for the device
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	OCL_CHECK(err);

	// Create a context  
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	OCL_CHECK(err);

	// Create a command queue 
	queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
	OCL_CHECK(err);

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &err);
	OCL_CHECK(err);

	// turn on optimization for kernel
	char *options="-cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros -cl-unsafe-math-optimizations -cl-finite-math-only";

	err = clBuildProgram(program, 1, &device_id, options, NULL, NULL);
	if(err != CL_SUCCESS)
		printCompilerOutput(program, device_id);
	OCL_CHECK(err);

	kernel[0] = clCreateKernel(program, "reduction_3a", &err);
	OCL_CHECK(err);

	kernel[1] = clCreateKernel(program, "reduction_3b", &err);
	OCL_CHECK(err);

	// memory on device
	cl_mem A_d   = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*N,  NULL, NULL);
	cl_mem intersum_d = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N,      NULL, NULL);
	cl_mem sum_d = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float),      NULL, NULL);

	// Initialize device memory
	err = clEnqueueWriteBuffer(queue, A_d, 	CL_TRUE, 0, sizeof(float)*N*N, 	A, 0, NULL, &event[0]); 
	OCL_CHECK(err);


	// kernel a
	size_t localsize[2];
	size_t globalsize[2];

	localsize[0] = 1;
	localsize[1] = TILE;

	globalsize[0]= N;
	globalsize[1]= TILE;

	err  = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &A_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &intersum_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 2, sizeof(float)*TILE, NULL);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 3, sizeof(int), &N);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 4, sizeof(int), &blks);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}


	// kernel b
	size_t k2_localsize;
	size_t k2_globalsize;

	k2_localsize = TILE;
	k2_globalsize= blks* TILE * 0.5;

	err  = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), &intersum_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[1], 1, sizeof(cl_mem), &sum_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[1], 2, sizeof(float)*TILE, NULL);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[1], 3, sizeof(int), &N);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[1], 4, sizeof(int), &blks);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clEnqueueNDRangeKernel(queue, kernel[0], 2, NULL, globalsize, localsize, 0, NULL, NULL);
	OCL_CHECK(err);

	err = clEnqueueNDRangeKernel(queue, kernel[1], 1, NULL, &k2_globalsize, &k2_localsize, 0, NULL, NULL);
	OCL_CHECK(err);







	clFinish(queue);

	clEnqueueReadBuffer(queue, sum_d, CL_TRUE, 0, sizeof(float), sum, 0, NULL , &event[1]);

	err = clWaitForEvents(1,&event[1]);
	OCL_CHECK(err);

	err = clGetEventProfilingInfo (event[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &gstart, NULL);
	OCL_CHECK(err);

	err = clGetEventProfilingInfo (event[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &gend, NULL);
	OCL_CHECK(err);

	gpuTime = (double)(gend -gstart)/1000000000.0;

	printf("oclTime = %lf (s)\n", gpuTime );

	check_1d_f(sum, 1);

#ifdef DEBUG
	//puts("C");
	//check_2d_f(C,N,N);

#endif


	// free
	clReleaseMemObject(A_d);	
	//clReleaseMemObject(C_d);
	clReleaseMemObject(sum_d);
	clReleaseMemObject(intersum_d);


	clReleaseProgram(program);
	clReleaseContext(context);
	clReleaseCommandQueue(queue);
	for(i=0;i<NumK;++i){
		clReleaseKernel(kernel[i]);
	}
	for(i=0;i<NumE;++i){
		clReleaseEvent(event[i]);
	}
	free(kernelSource);



	free(A);
	free(sum);

	return;
}



void run4(int N)
{
	puts("run 4\n");


	float *A;
	A = (float*)malloc(sizeof(float)*N*N);
	init_2d_f(A,N,N,1.f);

#ifdef DEBUG
	puts("A");
	check_2d_f(A,N,N);
#endif

	//float *C;
	//C = (float*)malloc(sizeof(float)*N*N);

	int blks = (N+TILE16-1)/TILE16;

	float *sum;
	sum = (float*)malloc(sizeof(float));

	int NumK = 1;
	int NumE = 2;

	int i;

	double gpuTime;
	cl_ulong gstart, gend;

	//------------------------------------------------
	//  OpenCL 
	//------------------------------------------------
	cl_int err;

	cl_platform_id platform;          // OpenCL platform
	cl_device_id device_id;           // device ID
	cl_context context;               // context
	cl_command_queue queue;           // command queue
	cl_program program;               // program

	cl_kernel *kernel = (cl_kernel*)malloc(sizeof(cl_kernel)*NumK);

	cl_event *event = (cl_event*)malloc(sizeof(cl_event)*NumE);    

	// read kernel file
	char *fileName = "kernel.cl";
	char *kernelSource;
	size_t size;
	FILE *fh = fopen(fileName, "rb");
	if(!fh) {
		printf("Error: Failed to open kernel file!\n");
		exit(1);
	}
	fseek(fh,0,SEEK_END);
	size=ftell(fh);
	fseek(fh,0,SEEK_SET);
	kernelSource = malloc(size+1);
	size_t result;
	result = fread(kernelSource,1,size,fh);
	if(result != size){ fputs("Reading error", stderr);exit(1);}
	kernelSource[size] = '\0';
	
	// Bind to platform
	err = clGetPlatformIDs(1, &platform, NULL);
	OCL_CHECK(err);

	// Get ID for the device
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	OCL_CHECK(err);

	// Create a context  
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	OCL_CHECK(err);

	// Create a command queue 
	queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
	OCL_CHECK(err);

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &err);
	OCL_CHECK(err);

	// turn on optimization for kernel
	char *options="-cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros -cl-unsafe-math-optimizations -cl-finite-math-only";

	err = clBuildProgram(program, 1, &device_id, options, NULL, NULL);
	if(err != CL_SUCCESS)
		printCompilerOutput(program, device_id);
	OCL_CHECK(err);

	kernel[0] = clCreateKernel(program, "reduction_4a", &err);
	OCL_CHECK(err);

	//kernel[1] = clCreateKernel(program, "reduction_3b", &err);
	//OCL_CHECK(err);

	// memory on device
	cl_mem A_d   = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*N,  NULL, NULL);
	cl_mem intersum_d = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*blks,      NULL, NULL);
	cl_mem sum_d = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float),      NULL, NULL);

	// Initialize device memory
	err = clEnqueueWriteBuffer(queue, A_d, 	CL_TRUE, 0, sizeof(float)*N*N, 	A, 0, NULL, &event[0]); 
	OCL_CHECK(err);


	// kernel a
	size_t localsize[2];
	size_t globalsize[2];

	localsize[0] = TILE16;
	localsize[1] = TILE16;

	globalsize[0]= blks*TILE16;
	globalsize[1]= TILE16;

	err  = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &A_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &intersum_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 2, sizeof(cl_mem), &sum_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 3, sizeof(float)*TILE16*TILE16, NULL);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 4, sizeof(int), &N);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 5, sizeof(int), &blks);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}


	err = clEnqueueNDRangeKernel(queue, kernel[0], 2, NULL, globalsize, localsize, 0, NULL, NULL);
	OCL_CHECK(err);


	clFinish(queue);

	clEnqueueReadBuffer(queue, sum_d, CL_TRUE, 0, sizeof(float), sum, 0, NULL , &event[1]);

	err = clWaitForEvents(1,&event[1]);
	OCL_CHECK(err);

	err = clGetEventProfilingInfo (event[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &gstart, NULL);
	OCL_CHECK(err);

	err = clGetEventProfilingInfo (event[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &gend, NULL);
	OCL_CHECK(err);

	gpuTime = (double)(gend -gstart)/1000000000.0;

	printf("oclTime = %lf (s)\n", gpuTime );

	check_1d_f(sum, 1);

#ifdef DEBUG
	//puts("C");
	//check_2d_f(C,N,N);

#endif


	// free
	clReleaseMemObject(A_d);	
	clReleaseMemObject(sum_d);
	clReleaseMemObject(intersum_d);


	clReleaseProgram(program);
	clReleaseContext(context);
	clReleaseCommandQueue(queue);
	for(i=0;i<NumK;++i){
		clReleaseKernel(kernel[i]);
	}
	for(i=0;i<NumE;++i){
		clReleaseEvent(event[i]);
	}
	free(kernelSource);



	free(A);
	free(sum);

	return;
}





//---------------
// use shared mem
//---------------
/*
void run_opt(int N, int T)
{

	puts("Optimized");

	float *A;
	A = (float*)malloc(sizeof(float)*N*T);
	init_2d_f(A,N,T,1.f);

#ifdef DEBUG
	puts("A");
	check_2d_f(A,N,T);
#endif

	float *B;
	B = (float*)malloc(sizeof(float)*T*N);
	init_2d_f(B,T,N,2.f);

#ifdef DEBUG
	puts("B");
	check_2d_f(B,T,N);
#endif

	float *C;
	C = (float*)malloc(sizeof(float)*N*N);


	int NumK = 1;
	int NumE = 2;

	int i;

	double gpuTime;
	cl_ulong gstart, gend;

	//------------------------------------------------
	//  OpenCL 
	//------------------------------------------------
	cl_int err;

	cl_platform_id platform;          // OpenCL platform
	cl_device_id device_id;           // device ID
	cl_context context;               // context
	cl_command_queue queue;           // command queue
	cl_program program;               // program

	cl_kernel *kernel = (cl_kernel*)malloc(sizeof(cl_kernel)*NumK);

	cl_event *event = (cl_event*)malloc(sizeof(cl_event)*NumE);    

	// read kernel file
	char *fileName = "mm.cl";
	char *kernelSource;
	size_t size;
	FILE *fh = fopen(fileName, "rb");
	if(!fh) {
		printf("Error: Failed to open kernel file!\n");
		exit(1);
	}
	fseek(fh,0,SEEK_END);
	size=ftell(fh);
	fseek(fh,0,SEEK_SET);
	kernelSource = malloc(size+1);
	size_t result;
	result = fread(kernelSource,1,size,fh);
	if(result != size){ fputs("Reading error", stderr);exit(1);}
	kernelSource[size] = '\0';
	
	// Bind to platform
	err = clGetPlatformIDs(1, &platform, NULL);
	OCL_CHECK(err);

	// Get ID for the device
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	OCL_CHECK(err);

	// Create a context  
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	OCL_CHECK(err);

	// Create a command queue 
	queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
	OCL_CHECK(err);

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &err);
	OCL_CHECK(err);

	// turn on optimization for kernel
	char *options="-cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros -cl-unsafe-math-optimizations -cl-finite-math-only";

	err = clBuildProgram(program, 1, &device_id, options, NULL, NULL);
	if(err != CL_SUCCESS)
		printCompilerOutput(program, device_id);
	OCL_CHECK(err);

	kernel[0] = clCreateKernel(program, "matrix_mul", &err);
	OCL_CHECK(err);


	// memory on device
	cl_mem A_d = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*T,  NULL, NULL);
	cl_mem B_d = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*T*N,  NULL, NULL);
	cl_mem C_d = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*N,  NULL, NULL);

	// Initialize device memory
	err = clEnqueueWriteBuffer(queue, A_d, 	CL_TRUE, 0, sizeof(float)*N*T, 	A, 0, NULL, NULL); 
	OCL_CHECK(err);

	err = clEnqueueWriteBuffer(queue, B_d, 	CL_TRUE, 0, sizeof(float)*T*N, 	B, 0, NULL, NULL); 
	OCL_CHECK(err);

	size_t localsize[2];
	size_t globalsize[2];

	localsize[0] = TILE;
	localsize[1] = TILE;

	globalsize[0] = ((N+TILE-1)/TILE)*16;
	globalsize[1] = ((N+TILE-1)/TILE)*16;


	err  = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &A_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &B_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 2, sizeof(cl_mem), &C_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 3, sizeof(float)*TILE*TILE, NULL);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 4, sizeof(float)*TILE*TILE, NULL);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 5, sizeof(int), &N);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 6, sizeof(int), &T);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err = clEnqueueNDRangeKernel(queue, kernel[0], 2, NULL, globalsize, localsize, 0, NULL, NULL);
	OCL_CHECK(err);

	clFinish(queue);

	clEnqueueReadBuffer(queue, C_d, CL_TRUE, 0, sizeof(float)*N*N, C, 0, NULL , NULL);

#ifdef DEBUG
	puts("C");
	check_2d_f(C,N,N);
#endif



	// free
	clReleaseMemObject(A_d);	
	clReleaseMemObject(B_d);
	clReleaseMemObject(C_d);


	clReleaseProgram(program);
	clReleaseContext(context);
	clReleaseCommandQueue(queue);
	for(i=0;i<NumK;++i){
		clReleaseKernel(kernel[i]);
	}
	for(i=0;i<NumE;++i){
		clReleaseEvent(event[i]);
	}
	free(kernelSource);



	free(A);
	free(B);
	free(C);

	return;
}
*/
