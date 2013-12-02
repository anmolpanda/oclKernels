//#pragma OPENCL EXTENSION all : enable
//#pragma OPENCL EXTENSION cl_amd_printf : enable

#define TILE 16 

__kernel void kernel_symmetrize(
		__global float *A, 
		const int blk_rows,
		const int width)
{
	// find the subdiagnoal blocks
	// for example,
	// 0 
	// 1	2
	// 3	4	5
	// 6	7	8	9

	// (0.0)
	// (1,0)	(1,1)
	// (2,0)	(2,1)	(2,2)
	// (3,0)	(3,1)	(3,2)	(3,3)

	// here , each block is [TILE][TILE]
	size_t lid_x = get_local_id(0);
	size_t lid_y = get_local_id(1);

	//size_t gid_y = get_global_id(1);
	//size_t gid_x = get_global_id(0);

	size_t bn = get_group_id(1); 
	
	int2 blkid;
	// find out the blocktag 
	int i,j;
	int upper,lower;
	for(i = 2, lower = 0, upper = 1 ; i <= (blk_rows+1) ; i++)
	{
		if( bn >= lower && bn < upper)
		{
			blkid.y = i-2; // rows
			blkid.x = bn - lower; // cols
			break;
		}
		else
		{
			lower = upper;
			upper = upper + i;
		}
	}

	// find out the block index, like blks(x,y)
	//if(lid_x == 0 && lid_y == 0)
	//{
	//	printf("(%d,%d)\n",blkid.y,blkid.x);
	//}

	// blkid.y : row
	// blkid.x : col
	// current blockid is (blkid.y, blkid.x)
	// then the symmetric (transposed) blockid is (blkid.x,blkid.y)

	
	// find the corresponding global thread index
	size_t gx,gy,gid;
	gx = blkid.x * TILE + lid_x;  // global column index
	gy = blkid.y * TILE + lid_y;
	gid = gy * width + gx;

	size_t gid_sym = gx * width + gy;

	//size_t gx_sym, gy_sym, gid_sym;
	//gx_sym = blkid.y * TILE + lid_x;
	//gy_sym = blkid.x * TILE + lid_y;
	//gid_sym = gy_sym * width + gx_sym;

	float a = A[gid];
	float b = A[gid_sym];
	if( a > b )
	{
		A[gid_sym] = a;
	}
	else
	{
		A[gid] = b;
	}

}


__kernel void kernel_symmetrize_iterative(
		__global float *A,        
		const int width)
{
	size_t gid = get_global_id(0);
	size_t lid = get_local_id(0);
	// DxD

	float a,b;
	// 

	if(gid < width)
	{
		int i;
		#pragma unroll
		for(i=gid;i<width;++i)
		{

			a = A[i*width + gid];        
			b = A[gid*width + i];        

			if(a>b){
				A[gid*width + i] = a;
			}else{
				A[i*width + gid] = b;
			}
		}
	}
}




