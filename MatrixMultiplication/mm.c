#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include <CL/opencl.h>

#include "../Utils/c_utils.h"
#include "../Utils/ocl_utils.h"

#define TILE 16

//-----------------------------
//    A(NxT)  *  B (TxN) = C
//-----------------------------

void run_naive(int N, int T);
void run_opt(int N, int T);


int main (int argc, char *argv[])
{
	if (argc !=  2){
		printf("Type ./mm N(datalen)\n");
		exit(1);
	}	

	int N = atoi(argv[1]); // row
	int T = N; // col

	//run_naive(N,T);

	run_opt(N,T);



	return  0;
}

void run_naive(int N, int T)
{
	puts("Naive");

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

	kernel[0] = clCreateKernel(program, "matrix_mul_naive", &err);
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

	localsize[0] = 16;
	localsize[1] = 16;

	globalsize[0] = ((N+15)/16)*16;
	globalsize[1] = ((N+15)/16)*16;

	err  = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &A_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err  = clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &B_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err  = clSetKernelArg(kernel[0], 2, sizeof(cl_mem), &C_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err  = clSetKernelArg(kernel[0], 3, sizeof(int), &N);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}
	err  = clSetKernelArg(kernel[0], 4, sizeof(int), &T);
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

//---------------
// use shared mem
//---------------

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
