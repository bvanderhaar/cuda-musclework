/**
 * blockAndThread.cu
 * description:	 	fill two arrays with blockId and threadId values	
 * notes:		compile with nvcc and parent code:
 * 			"nvcc blockAndThread.c blockAndThread.cu"
 * Program is similar to one that appears in Dr. Dobbs Journal.
 * The tutorial is available at:
 * http://www.ddj.com/hpc-high-performance-computing/207200659
 * Also used Andrew Bellenir's matrix multiplication program
**/

#include <stdio.h>
#include <stdlib.h>

/* 
 * In CUDA it is necessary to define block sizes
 * The grid of data that will be worked on is divided into blocks
 */
#define BLOCK_SIZE 8 
/**
 * The function that will be executed in each stream processors
 * The __global__ directive identifies this function as being
 * an executable kernel on the CUDA device.
 * All kernesl must be declared with a return type void 
 */ 
__global__ void cu_fillArray(int *block_d,int *thread_d){
        int x;
	/* blockIdx.x is a built-in variable in CUDA
           that returns the blockId in the x axis.
           threadIdx.x is another built-in variable in CUDA
           that returns the threadId in the x axis
           of the thread that is being executed by the
           stream processor accessing this particular block
        */
	x=blockIdx.x*BLOCK_SIZE+threadIdx.x;
	block_d[x] = blockIdx.x;
	thread_d[x] = threadIdx.x;
}

/**
 * This function is called from the host computer.
 * It calls the function that is executed on the GPU.
 * Recall that:
 *  The host computer and the GPU have separate memories
 *  Hence it is necessary to 
 *    - Allocate memory in the GPU 
 *    - Copy the variables that will be operated on from host 
 *      memory to the corresponding variables in the GPU
 *    - Describe the configuration of the grid and the block size
 *    - Call the kernel, the code that will be executed on the GPU
 *    - Once the kernel has finished executing, copy the results
 *      back from GPU memory to host memory
 */
extern "C" void fillArray(int *block,int *thread,int arraySize){
	//block_d and thread_d are the GPU counterparts of the arrays that exist in the host memory 
	int *block_d;
	int *thread_d;
	cudaError_t result;
	//allocate memory on device
	// cudaMalloc allocates space in the memory of the GPU
	result = cudaMalloc((void**)&block_d,sizeof(int)*arraySize);
	if (result != cudaSuccess) {
		printf("cudaMalloc - block_d - failed\n");
		exit(1);
	}
	result = cudaMalloc((void**)&thread_d,sizeof(int)*arraySize);
	if (result != cudaSuccess) {
		printf("cudaMalloc - thread_d - failed\n");
		exit(1);
	}
	//copy the arrays into the variable array_d in the device
	result = cudaMemcpy(block_d,block,sizeof(int)*arraySize,cudaMemcpyHostToDevice);
	if (result != cudaSuccess) {
		printf("cudaMemcpy - host-GPU - block - failed\n");
		exit(1);
	}
	result = cudaMemcpy(thread_d,thread,sizeof(int)*arraySize,cudaMemcpyHostToDevice);
	if (result != cudaSuccess) {
		printf("cudaMemcpy - host-GPU - thread - failed\n");
		exit(1);
	}

	//execution configuration...
	// Indicate the dimension of the block
	dim3 dimblock(BLOCK_SIZE);
	// Indicate the dimension of the grid in blocks 
	dim3 dimgrid(arraySize/BLOCK_SIZE);
	//actual computation: Call the kernel
	cu_fillArray<<<dimgrid,dimblock>>>(block_d,thread_d);
	// read results back:
	// Copy the results from the memory in the GPU back to the memory on the host
	result = cudaMemcpy(block,block_d,sizeof(int)*arraySize,cudaMemcpyDeviceToHost);
	if (result != cudaSuccess) {
		printf("cudaMemcpy - GPU-host - block - failed\n");
		exit(1);
	}
	result = cudaMemcpy(thread,thread_d,sizeof(int)*arraySize,cudaMemcpyDeviceToHost);
	if (result != cudaSuccess) {
		printf("cudaMemcpy - GPU-host - thread - failed\n");
		exit(1);
	}
	// Release the memory on the GPU 
	cudaFree(block_d);
	cudaFree(thread_d);
}

