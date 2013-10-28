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

void run1(int N, char *fileName); // naive 

void run2(int N, char *fileName); // optimized 



int main (int argc, char *argv[])
{
	if (argc !=  3){
		printf("Type ./Program N(square matrix width) kernelfile.cl \n");
		exit(1);
	}	

	int N = atoi(argv[1]); // row

	// run1(N, argv[2]);

	 run2(N, argv[2]);

	return  0;
}



//-----------------------------------------------------------
// run 2
//-----------------------------------------------------------

void run2(int N, char *fileName)
{
	puts("Optimized Matrix Transpose");

	int i,j;

	float *A;
	A = (float*)malloc(sizeof(float)*N*N);

	for( i = 0; i < N ; ++i )
	{
		for( j = 0; j < N ; ++j )
		{
			A[i*N + j] = j;	
		}
	}

	float *At;
	At = (float*)malloc(sizeof(float)*N*N);


#ifdef DEBUG
	puts("A");
	check_2d_f(A,N,N);
#endif

	int NumK = 1;
	int NumE = 1;

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
	//char *fileName = "transpose_kernel.cl";
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



#ifdef SAVEBIN
	// Calculate size of binaries 
	size_t binary_size;
	err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_size, NULL);
	OCL_CHECK(err);

	//printf("binary size = %ld\n", binary_size);

	unsigned char* bin;
	bin = (unsigned char*)malloc(sizeof(unsigned char)*binary_size);


	err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*) , &bin, NULL);
	//printf("err = %d\n", err);

	OCL_CHECK(err);

	//puts("save binaries");

	// Print the binary out to the output file
	fh = fopen("kernel_run2.bin", "wb");
	fwrite(bin, 1, binary_size, fh);
	fclose(fh);

	puts("done save binaries");

#endif




	kernel[0] = clCreateKernel(program, "transpose_2", &err);
	OCL_CHECK(err);

	// memory on device
	cl_mem A_d    = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*N,  NULL, NULL);
	cl_mem At_d   = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*N,  NULL, NULL);

	// Initialize device memory
	err = clEnqueueWriteBuffer(queue, A_d, 	CL_TRUE, 0, sizeof(float)*N*N, 	A, 0, NULL , &event[0]); 
	OCL_CHECK(err);

	size_t localsize[2];
	size_t globalsize[2];

	localsize[0] = TILE; 
	localsize[1] = TILE;

	// each workitem in [TILE][TILE] will execute [elePerThread1Dim][elePerThread1Dim]
	
	//globalsize[0] = N;
	//globalsize[1] = N;

	globalsize[0] = N/2;
	globalsize[1] = N/2;

	err  = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &A_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &At_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 2, sizeof(float)*TILE*TILE*4, NULL);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}


	err = clEnqueueNDRangeKernel(queue, kernel[0], 2, NULL, globalsize, localsize, 0, NULL, &event[1]);
	OCL_CHECK(err);

	clFinish(queue);

	clEnqueueReadBuffer(queue, At_d, CL_TRUE, 0, sizeof(float)*N*N, At, 0, NULL , &event[2]);

	err = clWaitForEvents(1,&event[2]);
	OCL_CHECK(err);

	err = clGetEventProfilingInfo (event[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &gstart, NULL);
	OCL_CHECK(err);

	err = clGetEventProfilingInfo (event[2], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &gend, NULL);
	OCL_CHECK(err);

	gpuTime = (double)(gend -gstart)/1000000000.0;



	//check_1d_f(sum, blks+1);

#ifdef DEBUG
	puts("Transpose (A)");
	check_2d_f(At,N,N);

#endif

	printf("oclTime = %lf (s)\n", gpuTime );

	// free
	clReleaseMemObject(A_d);	
	clReleaseMemObject(At_d);	


	// check
	int flag = 1;
	for(i=0;i<N;++i){
		for(j=0;j<N;++j){
			if(A[i*N+j] != At[j*N+i])		
			{
				flag  = 0;
				break;
			}
		}
	}
	if( flag == 0 )
	{
		puts("Bugs! Check program.");
	}else{
		puts("Succeed!");	
	}



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


#ifdef SAVEBIN
	free(bin);
#endif



	free(A);
	free(At);

	return;
}




//-----------------------------------------------------------
// run 1
//-----------------------------------------------------------

void run1(int N, char *fileName)
{
	puts("Transpose Naive\n");

	int i,j;

	float *A;
	A = (float*)malloc(sizeof(float)*N*N);

	for( i = 0; i < N ; ++i )
	{
		for( j = 0; j < N ; ++j )
		{
			A[i*N + j] = j;	
		}
	}

	float *At;
	At = (float*)malloc(sizeof(float)*N*N);


#ifdef DEBUG
	puts("A");
	check_2d_f(A,N,N);
#endif

	int NumK = 1;
	int NumE = 1;

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
	//char *fileName = "transpose_kernel.cl";
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

#ifdef SAVEBIN
	// Calculate size of binaries 
	size_t binary_size;
	err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_size, NULL);
	OCL_CHECK(err);

	unsigned char* bin;
	bin = (unsigned char*)malloc(sizeof(unsigned char)*binary_size);

	err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*), &bin, NULL);
	OCL_CHECK(err);

	// Print the binary out to the output file
	fh = fopen("kernel_run1.bin", "wb");
	fwrite(bin, 1, binary_size, fh);
	fclose(fh);

#endif


	kernel[0] = clCreateKernel(program, "transpose_1", &err);
	OCL_CHECK(err);


	// memory on device
	cl_mem A_d    = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*N,  NULL, NULL);
	cl_mem At_d   = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(float)*N*N,  NULL, NULL);

	// Initialize device memory
	err = clEnqueueWriteBuffer(queue, A_d, 	CL_TRUE, 0, sizeof(float)*N*N, 	A, 0, NULL , NULL); 
	OCL_CHECK(err);

	size_t localsize[2];
	size_t globalsize[2];

	localsize[0] = TILE;
	localsize[1] = TILE;

	int blks = ( N+TILE-1)/TILE;

	globalsize[0] = blks * TILE;
	globalsize[1] = blks * TILE;

	err  = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &A_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &At_d);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}

	err  = clSetKernelArg(kernel[0], 2, sizeof(int), &N);
	if(err != 0) { printf("%d\n",err); OCL_CHECK(err); exit(1);}


	err = clEnqueueNDRangeKernel(queue, kernel[0], 2, NULL, globalsize, localsize, 0, NULL, &event[0]);
	OCL_CHECK(err);

	clFinish(queue);

	clEnqueueReadBuffer(queue, At_d, CL_TRUE, 0, sizeof(float)*N*N, At, 0, NULL , NULL );

	err = clWaitForEvents(1,&event[0]);
	OCL_CHECK(err);

	err = clGetEventProfilingInfo (event[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &gstart, NULL);
	OCL_CHECK(err);

	err = clGetEventProfilingInfo (event[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &gend, NULL);
	OCL_CHECK(err);

	gpuTime = (double)(gend -gstart)/1000000000.0;



	//check_1d_f(sum, blks+1);

#ifdef DEBUG
	puts("Transpose (A)");
	check_2d_f(At,N,N);

#endif

	printf("oclTime = %lf (s)\n", gpuTime );

	// free
	clReleaseMemObject(A_d);	
	clReleaseMemObject(At_d);	


	// check
	int flag = 1;
	for(i=0;i<N;++i){
		for(j=0;j<N;++j){
			if(A[i*N+j] != At[j*N+i])		
			{
				flag  = 0;
				break;
			}
		}
	}
	if( flag == 0 )
	{
		puts("Bugs! Check program.");
	}else{
		puts("Succeed!");	
	}



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


#ifdef SAVEBIN
	free(bin);
#endif

	free(A);
	free(At);

	return;
}




