//#pragma OPENCL EXTENSION all : enable


__kernel void matrix_mul(
			 __global float *A, 
			 __global float *B, 
			 __global float *C, 
			 const int rA, 
			 const int cA,
			 const int cB)
{

	int tx = get_global_id(0); 
	int ty = get_global_id(1);

	if ( tx < rA && ty < cB )
	{
		float value = 0.f;
		int k;
		for ( k = 0; k < cA; ++k) 
		{
			float elementA = A[tx * cA + k];
			float elementB = B[k * cB + ty];
			value += elementA * elementB;
		}

		C[tx * cB + ty] = value;
	}
}


