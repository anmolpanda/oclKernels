//#pragma OPENCL EXTENSION all : enable
//#pragma OPENCL EXTENSION cl_amd_printf : enable

#define TILE 16

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


__kernel void transpose_2(
		__global float *A, 
		__global float *At, 
		const int N)
{
	// square matrix[N][N]

	size_t gx = get_group_id(0);
	size_t gy = get_group_id(1);

	size_t blks_x = get_num_group(0);	

	// reshuffle blocks
	size_t giy = gx;
	size_t gix = (gx + gy)%blks_x;

	size_t lix = get_local_id(0);
	size_t liy = get_local_id(1);

	// use reshuffled blks to index the read data
	size_t ix = gix * TILE + lix; 
	size_t iy = giy * TILE + liy; 

	size_t index_in = ix + iy * N;

	
}
