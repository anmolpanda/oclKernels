//#pragma OPENCL EXTENSION all : enable


__kernel void matrix_mul(
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


