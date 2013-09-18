#include <stdlib.h>
#include <stdio.h>
#include <CL/opencl.h>

#include "ocl_utils.c"

int main(int argc, char *argv[])
{
	int i,j;

	cl_int err;	

	cl_platform_id *platform_id;
	cl_device_id *device_id;

	cl_uint numPlatforms; 
	cl_uint numDev; 
	cl_uint devCU;

	size_t devGroup;

	char pbuf[100];

	// query platform info
	err = clGetPlatformIDs(0,NULL,&numPlatforms);
	OCL_CHECK(err);
	printf("\nnumPlatforms = %d\n\n", numPlatforms);

	if(numPlatforms==0){
		printf("Error: no platform is found!\n");
		exit(1);
	}


	platform_id = (cl_platform_id*)malloc(sizeof(cl_platform_id)*numPlatforms);

	for(i=0;i<numPlatforms;++i){
		err = clGetPlatformIDs(1,&platform_id[i], NULL);// choose the 1st platform
		OCL_CHECK(err);

		err = clGetPlatformInfo(platform_id[i],CL_PLATFORM_NAME,sizeof(pbuf),pbuf,NULL);
		OCL_CHECK(err);

		printf("platform[0] name: %s\n",pbuf);

		// query device info
		err = clGetDeviceIDs(platform_id[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDev);
		OCL_CHECK(err);

		printf("platform device number: %d\n",numDev);

		device_id = (cl_device_id*)malloc(sizeof(cl_device_id)*numDev);

		err = clGetDeviceIDs(platform_id[i],CL_DEVICE_TYPE_ALL,numDev,device_id,NULL);
		OCL_CHECK(err);

		for(j=0;j<numDev;++j){
			err = clGetDeviceInfo(device_id[j],CL_DEVICE_NAME,sizeof(pbuf),pbuf,NULL);OCL_CHECK(err);
			printf("\n--- device %d:---\n",j);
			printf("name: %s\n",pbuf);
			//err = clGetDeviceInfo(device_id[i],CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV,sizeof(cl_uint),(void*)&devMajor,NULL);OCL_CHECK(err);
			//err = clGetDeviceInfo(device_id[i],CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV,sizeof(cl_uint),(void*)&devMinor,NULL);OCL_CHECK(err);
			//printf("Capability: %d.%d\n",devMajor,devMinor);
			err = clGetDeviceInfo(device_id[j],CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),(void*)&devCU,NULL);OCL_CHECK(err);
			printf("computer units: %d\n",devCU);
			err = clGetDeviceInfo(device_id[j],CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(size_t),(void*)&devGroup,NULL);OCL_CHECK(err);
			printf("work group size: %ld\n", devGroup);
		}

	}
	printf("\n");

	return 0;
}

