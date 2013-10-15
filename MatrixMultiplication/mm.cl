//#pragma OPENCL EXTENSION all : enable
#define TILE 16

__kernel void matrix_mul_naive(
		__global float *A, 
		__global float *B, 
		__global float *C, 
		const int N, 
		const int T)
{

	int tx = get_global_id(0); 
	int ty = get_global_id(1);

	if ( tx < N && ty < N)
	{
		float value = 0.f;
		int k;
		for ( k = 0; k < T; ++k) 
		{
			float elementA = A[tx * T + k];
			float elementB = B[k  * N + ty];
			value += elementA * elementB;
		}

		C[tx * N + ty] = value;
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

	float sum;

	int m,k ;

	for ( m = 0; m < T/TILE ; ++m)
	{
		sma[tx * TILE + ty] = A[Row * T + m * TILE + ty];	
		smb[tx * TILE + ty] = B[(m * TILE + tx) * T + Col];	
	
		barrier(CLK_LOCAL_MEM_FENCE);

		sum = 0.f;	

		for ( k = 0; k < TILE ; ++ k) 
		{
			sum += sma[tx * TILE + k] * smb[k * TILE + ty];
		}
	
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	C[Row * N + Col] = sum;

}
