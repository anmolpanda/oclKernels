//#pragma OPENCL EXTENSION all : enable
//#pragma OPENCL EXTENSION cl_amd_printf : enable


__kernel void transpose_1(
		__global float *A, 
		__global float *At, 
		const int N)
{
	size_t gx = get_global_id(0); 
	size_t gy = get_global_id(1); 

	if ( gx < N && gy < N )
	{
		At[gy * N + gx] = A[gx * N + gy];
	}

}


