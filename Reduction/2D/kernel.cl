//#pragma OPENCL EXTENSION all : enable
#pragma OPENCL EXTENSION cl_amd_printf : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

/*
inline void AtomicAdd(volatile __global float *source, const float operand) {
	union {
		unsigned int intVal;
		float floatVal;
	} newVal;

	union {
		unsigned int intVal;
		float floatVal;
	} prevVal;

	do {
		prevVal.floatVal = *source;
		newVal.floatVal = prevVal.floatVal + operand;
	} while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}
*/

__kernel void reduction_1(
		__global float *A, 
		__global float *C, 
		__global float *sum,
		__local  volatile float *sm,
		const int N)
{
	size_t gid = get_global_id(0); 
	size_t lid = get_local_id(0); 

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

	barrier(CLK_LOCAL_MEM_FENCE);

	if ( lid < 32) {
		sm[lid] += sm[lid + 32];
		sm[lid] += sm[lid + 16];
		sm[lid] += sm[lid +  8];
		sm[lid] += sm[lid +  4];
		sm[lid] += sm[lid +  2];
		sm[lid] += sm[lid +  1];
	}

	if(lid == 0) {
		//printf("%f\n", sm[0]);
		//printf("sum[0] = %f\n", sum[0]);
		AtomicAdd(sum,sm[0]);
		//float x = sm[0];
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
