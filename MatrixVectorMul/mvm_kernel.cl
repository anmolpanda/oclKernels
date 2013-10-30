//#pragma OPENCL EXTENSION all : enable
//#pragma OPENCL EXTENSION cl_amd_printf : enable


#define TILE 16

__kernel void mv_1(
		__global float *A, 
		__global float *B, 
		__global float *C, 
		const int N)
{
	size_t gid = get_global_id(0);  // row

	int i;
	float sum = 0.f;
	for(i = 0 ; i< N; ++i ) // columns
	{
		sum += A[gid * N  + i]	* B[i];
	}

	C[gid] = sum;
}

__kernel void mv_2(
		__global float *A, 
		__global float *B, 
		__global float *C, 
		__local  float *lds,
		const int N)
{
	// lds[16][16]

	// hint: fetch B to local	
	size_t gx = get_global_id(0); // col  
	size_t gy = get_global_id(1); // row 

	size_t lx = get_local_id(0); // col  
	size_t ly = get_local_id(1); // row 
	
	size_t m =  N/TILE;

	int i;
	lds[ly*TILE + lx] = 0.0;
	for(i = 0 ; i<m ; ++i){
		lds[ly*TILE + lx]	 += A[gy * N + lx + i*TILE] * B[lx + i*TILE];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// sum across rows
	if( gx == 0)
	{
		int start = ly * TILE;
		C[gy] = lds[start] 		+ lds[start + 1] + lds[start + 2] 	+ lds[start + 3]  + lds[start + 4]  + lds[start + 5]  + lds[start + 6]  + lds[start + 7] + 
				lds[start + 8] 	+ lds[start + 9] + lds[start + 10]	+ lds[start + 11] + lds[start + 12] + lds[start + 13] + lds[start + 14] + lds[start + 15]; 
	}

}

// can we improve more?






