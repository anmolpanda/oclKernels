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

// each workitem works on [2x2] matrix elements 
__kernel void transpose_2(
		__global float *A, 
		__global float *At, 
		__local  float *lds )
{
	// square matrix
 	size_t N  = get_global_size(0);

	size_t gx = get_group_id(0);
	size_t gy = get_group_id(1);

	size_t blks_x = get_num_groups(0);	

	// reshuffle blocks
	size_t giy = gx;
	size_t gix = (gx + gy)%blks_x;

	size_t lix = get_local_id(0);
	size_t liy = get_local_id(1);

	// use reshuffled blks to index the reading data
	size_t ix = gix * TILE * 2  + lix; 
	size_t iy = giy * TILE * 2  + liy; 


	size_t index_in00 = ix 			+ iy * N ;        // [iy][ix]
	size_t index_in01 = (ix + TILE) + iy * N ; 		  // [iy][ix + TILE]
	size_t index_in10 = ix 			+ (iy+TILE) * N ; // [iy+TILE][ix]
	size_t index_in11 = (ix + TILE) + (iy+TILE) * N ; // [iy+TILE][ix+TILE]

	// copy from global memory to LDS
	size_t ind = liy * TILE + lix; //[liy][lix]

	lds[ind]					=	A[index_in00];	//[liy][lix]
	lds[ind + TILE]				=	A[index_in01];  //[liy][lix + TILE]
	lds[ind + TILE * TILE]		=	A[index_in10];  //[liy+TILE][lix]
	lds[ind + (TILE+1)* TILE]	=	A[index_in11];  //[liy+TILE][lix+TILE]

	barrier(CLK_LOCAL_MEM_FENCE);
	
	ix = giy * TILE * 2  + lix;
	iy = gix * TILE * 2  + liy;

	// transpose the index inside LDS
	ind = lix * TILE  + liy; // [lix][liy]

	int index_out00 = ix  		+ iy * N ;			// [iy][ix]
	int index_out01 = (ix+TILE) + iy * N ;			// [iy][ix + TILE]
	int index_out10 = ix  		+ (iy + TILE)* N ;	// [iy+TILE][ix]
	int index_out11 = (ix+TILE) + (iy + TILE)* N ;	// [iy+TILE][ix+TILE]

	At[index_out00]			= lds[ind];
	At[index_out01]			= lds[ind + TILE];
	At[index_out10]			= lds[ind + TILE * TILE];
	At[index_out11]			= lds[ind + (TILE+1) * TILE];

}
