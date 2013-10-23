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
		__local  float *lds, 
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

	// use reshuffled blks to index the reading data
	size_t ix = gix * TILE + lix; 
	size_t iy = giy * TILE + liy; 

	size_t index_in = ix + iy * N * 4;

	// copy from global memory to LDS
	size_t ind = liy * TILE * 4 + lix;

	lds[ind]			=	A[index_in];
	lds[ind + TILE]		=	A[index_in + N];
	lds[ind + TILE * 2]	=	A[index_in + N * 2];
	lds[ind + TILE * 3]	=	A[index_in + N * 3];

	barrier(CLK_LOCAL_MEM_FENCE);
	
	ix = giy * TILE + lix;
	iy = gix * TILE + liy;

	int index_out = ix  + iy * N * 4;

	// transpose the index inside LDS
	ind = liy + lix * TILE * 4; 

	At[index_out]			= lds[ind];
	At[index_out + N]		= lds[ind + TILE];
	At[index_out + N * 2]	= lds[ind + TILE * 2];
	At[index_out + N * 3]	= lds[ind + TILE * 3];


}
