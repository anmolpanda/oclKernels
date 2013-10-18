//#pragma OPENCL EXTENSION all : enable
#pragma OPENCL EXTENSION cl_amd_printf : enable


#define TILE 64
#define TILE16 16

__kernel void reduction_1(
		__global float *A, 
		__global float *sum,
		__local  volatile float *sm,
		const int N,
		const int blk)
{
	size_t gid = get_global_id(0); 
	size_t lid = get_local_id(0); 
	size_t blk_id =  get_group_id(0);
	int k;
	float value;

	if ( gid < N )
	{
		value = 0.f;
		for ( k = 0; k < N; ++k) 
		{
			value += A[gid*N + k];
		}

		//printf("value[%d] = %f\n", lid, value);
		sm[lid] = value;
		//printf("sm[%d] = %f\n", lid, sm[lid]);
	}else {
		sm[lid] = 0.f;

	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if ( gid < N )
	{
		if ( lid < 32) 
		{
			sm[lid] += sm[lid + 32];
			sm[lid] += sm[lid + 16];
			sm[lid] += sm[lid +  8];
			sm[lid] += sm[lid +  4];
			sm[lid] += sm[lid +  2];
			sm[lid] += sm[lid +  1];
		}

		if(lid == 0) 
		{
			//printf("%f\n", sm[0]);
			//printf("sum[0] = %f\n", sum[lid]);
			sum[blk_id] = sm[0];
		}
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if( gid == 0 ) 
	{
		// accumulate the first blk samples	
		value = 0.f;
		for ( k = 0 ; k < blk; ++k ) 
		{
			value += sum[k];	
		}
		sum[blk] = value;
	}

}



__kernel void reduction_2a(
		__global float *A, 
		__global float *sum,
		__local  volatile float *sm,
		const int N,
		const int blk)
{
	size_t gid = get_global_id(0); 
	size_t lid = get_local_id(0); 
	size_t blk_id =  get_group_id(0);
	int k;
	float value;

	if ( gid < N )
	{
		value = 0.f;
		for ( k = 0; k < N; ++k) 
		{
			value += A[gid*N + k];
		}

		sm[lid] = value;
	}else {
		sm[lid] = 0.f;

	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if ( gid < N )
	{
		if ( lid < 32) 
		{
			sm[lid] += sm[lid + 32];
			sm[lid] += sm[lid + 16];
			sm[lid] += sm[lid +  8];
			sm[lid] += sm[lid +  4];
			sm[lid] += sm[lid +  2];
			sm[lid] += sm[lid +  1];
		}

		if(lid == 0) 
		{
			//printf("%f\n", sm[0]);
			//printf("sum[0] = %f\n", sum[lid]);
			sum[blk_id] = sm[0];
		}
	}

}

__kernel void reduction_2b(
		__global float *sum,
		__local  volatile float *sm,
		const int N,
		const int blk)
{
	size_t gid = get_global_id(0); 
	size_t lid = get_local_id(0); 
	size_t blk_id =  get_group_id(0);

	if( gid < blk ) 
	{
		sm[lid] = sum[gid];
	}else{
		sm[lid] = 0.f; 
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if ( lid < 32) 
	{
		sm[lid] += sm[lid + 32];
		sm[lid] += sm[lid + 16];
		sm[lid] += sm[lid +  8];
		sm[lid] += sm[lid +  4];
		sm[lid] += sm[lid +  2];
		sm[lid] += sm[lid +  1];
	}

	if(lid == 0) 
	{
		sum[blk] = sm[0];
	}
}


__kernel void reduction_3a(
		__global float *A, 
		__global float *intersum,
		__local  volatile float *sm, // TILE
		const int N,
		const int blks)
{
	int gx = get_global_id(0); // row

	//int gy = get_global_id(1); // col
	//int bx = get_group_id(0);
	//int by = get_group_id(1);
	//int tx = get_local_id(0);

	int ty = get_local_id(1);

	int m;	
	float v = 0.f;

	#pragma unroll
	for ( m = 0; m < blks; ++m)
	{
		v += A[gx * N + m * TILE + ty];	
	}

	sm[ty] = v;

	barrier(CLK_LOCAL_MEM_FENCE);

	if ( ty < 32) 
	{
		sm[ty] += sm[ty + 32];
		sm[ty] += sm[ty + 16];
		sm[ty] += sm[ty +  8];
		sm[ty] += sm[ty +  4];
		sm[ty] += sm[ty +  2];
		sm[ty] += sm[ty +  1];
	}

	if(ty == 0) 
	{
		intersum[gx] = sm[0];
	}

}

__kernel void reduction_3b(
		__global float *intersum,
		__global float *sum,
		__local  volatile float *sm, // 64 
		const int N,
		const int blks)
{
	int gid = get_global_id(0); 
	int lid = get_local_id(0);
	int blkid = get_group_id(0);
	int offset = blkid * TILE;

	// read with stride of TILE
	intersum[gid+offset] += intersum[gid+offset+TILE];

	// copy to shared memory
	sm[lid] = intersum[gid+offset];

	barrier(CLK_LOCAL_MEM_FENCE);

	// reduction
	if ( lid < 32) 
	{
		sm[lid] += sm[lid + 32];
		sm[lid] += sm[lid + 16];
		sm[lid] += sm[lid +  8];
		sm[lid] += sm[lid +  4];
		sm[lid] += sm[lid +  2];
		sm[lid] += sm[lid +  1];
	}

	if( lid == 0) 
	{
		intersum[offset*2] = sm[0];
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if(gid == 0){
		int k;
		float v=0.f;
		#pragma unroll
		for ( k = 0 ; k < blks; ++k)
		{
			v += intersum[k*TILE*2];	
		}
		sum[0] = v;
	}

}


__kernel void reduction_4a(
		__global float *A, 
		__global float *intersum,
		__global float *sum,
		__local  volatile float *sm, // 256 
		const int N,
		const int blks)
{
	int gx = get_global_id(0); // row
	int gy = get_global_id(1); // col

	int bx = get_group_id(0);
	//int by = get_group_id(1);

	int tx = get_local_id(0);
	int ty = get_local_id(1);


	int tid =  tx * TILE16 + ty;


	// loop through columns		
	int m;	
	float v = 0.f;

	#pragma unroll
	for ( m = 0; m < blks; ++m)
	{
		v += A[gx * N + m * TILE16 + gy];	
	}

	sm[tid] = v;

	barrier(CLK_LOCAL_MEM_FENCE);

	if(tid < 128){
		sm[tid] += sm[tid + 128];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if(tid < 64){
		sm[tid] += sm[tid + 64];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if(tid < 32){
		sm[tid] += sm[tid + 32];
	}
	barrier(CLK_LOCAL_MEM_FENCE);


	if(tid < 16){
		sm[tid] += sm[tid + 16];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
/*
	if(tid < 8){
		sm[tid] += sm[tid + 8];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if(tid < 4){
		sm[tid] += sm[tid + 4];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if(tid < 2){
		sm[tid] += sm[tid + 2];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if(tid < 1){
		sm[tid] += sm[tid + 1];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
*/


/*
	if ( tid < 32) 
	{
		sm[tid] += sm[tid + 32];
		sm[tid] += sm[tid + 16];
		sm[tid] += sm[tid +  8];
		sm[tid] += sm[tid +  4];
		sm[tid] += sm[tid +  2];
		sm[tid] += sm[tid +  1];
	}
*/

	// southern island 7970M
/*
	if ( tid < 8) 
	{
		sm[tid] += sm[tid +  8];
		sm[tid] += sm[tid +  4];
		sm[tid] += sm[tid +  2];
		sm[tid] += sm[tid +  1];
	}
*/
	if(tid == 0) 
	{
		intersum[bx] = sm[0] + sm[1] + sm[2]  + sm[3]  + sm[4]  + sm[5]  + sm[6]  + sm[7] 
			         + sm[8] + sm[9] + sm[10] + sm[11] + sm[12] + sm[13] + sm[14] + sm[15];
		//printf("blk %d, sum = %f\n", bx, intersum[bx]);
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if( gx == 0 && gy == 0)
	{
		v = 0.f;
		#pragma unroll
		for ( m = 0; m < blks; ++m)
		{
			v += intersum[m];	
		}
		sum[0] = v;
	}

}





