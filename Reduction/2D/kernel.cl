//#pragma OPENCL EXTENSION all : enable
#define TILE 16

__kernel void reduction_1(
		__global float *A, 
		__global float *C, 
		__local  float *sm,
		const int N)
{
	size_t gid = get_global_id(0); 

	if ( gid < N )
	{
		float value = 0.f;
		int k;
		for ( k = 0; k < N; ++k) 
		{
			value += A[gid*N + k];
		}

		sm[lid] = value;
	}else {
		sm[lid] = 0.f;

	}

	// reductin on shared memory 
	if( lid <128 ){
		sm[lid] = sm[lid + 128];	
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if( lid < 64) {
		sm[lid] = sm[lid + 64];	
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if ( lid < 32) {
		volatile __local float *sm1 = sm;
		sm1[lid] += sm1[lid + 32];
		sm1[lid] += sm1[lid + 16];
		sm1[lid] += sm1[lid +  8];
		sm1[lid] += sm1[lid +  4];
		sm1[lid] += sm1[lid +  2];
		sm1[lid] += sm1[lid +  1];
	}

	if(lid == 0) {
		// atomic float add
		atomic_add(&glb_sum[0],sm[0]);
	}

}


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
