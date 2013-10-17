//#pragma OPENCL EXTENSION all : enable
#pragma OPENCL EXTENSION cl_amd_printf : enable


#define TILE 64

__kernel void reduction_1(
		__global float *A, 
		//__global float *C, 
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
		// __global float *C, 
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







/*
   __kernel void matrix_mul(
   __global float *A, 
   __global float *B, 
   __global float *C, 
   __local  float *sma, /// 16 x 16
   __local  float *smb,
   const int N, 
   const int T)
   {
// A [Row][m*TILE+ty]
// B [m*TILE+tx][Col] 
int gx = get_global_id(0); 
int gy = get_global_id(1);

int bx = get_group_id(0);
int by = get_group_id(1);

int tx = get_local_id(0);
int ty = get_local_id(1);

int Row =  bx * TILE + tx;
int Col =  by * TILE + ty;

float sum = 0.f;	

int m,k ;

for ( m = 0; m < T/TILE ; ++m)
{
sma[tx * TILE + ty] = A[Row * T + m * TILE + ty];	
smb[tx * TILE + ty] = B[(m * TILE + tx) * T + Col];	

barrier(CLK_LOCAL_MEM_FENCE);


for ( k = 0; k < TILE ; ++ k) 
{
sum += sma[tx * TILE + k] * smb[k * TILE + ty];
}

barrier(CLK_LOCAL_MEM_FENCE);
}

C[Row * N + Col] = sum;

}
 */
