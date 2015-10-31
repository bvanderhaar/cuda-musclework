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
__global__ void cu_dotProduct(int *block_d, int *thread_d) {
  int x;
  /* blockIdx.x is a built-in variable in CUDA
     that returns the blockId in the x axis.
     threadIdx.x is another built-in variable in CUDA
     that returns the threadId in the x axis
     of the thread that is being executed by the
     stream processor accessing this particular block
  */
  x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  block_d[x] = blockIdx.x;
  thread_d[x] = threadIdx.x;
}

// Called from driver program.  Handles running GPU calculation
extern "C" void gpu_dotProduct(int *array, int arraySize) {
  // a_d is the GPU counterpart of the array that exists in host memory
  int *array_d;
  cudaError_t result;

  // allocate space in the device
  result = cudaMalloc((void **)&array_d, sizeof(int) * arraySize);
  if (result != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed.");
    exit(1);
  }

  // copy the array from host to array_d in the device
  result = cudaMemcpy(array_d, array, sizeof(int) * arraySize,
                      cudaMemcpyHostToDevice);
  if (result != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed.");
    exit(1);
  }

  // set execution configuration
  dim3 dimblock(BLOCK_SIZE);
  dim3 dimgrid(arraySize / BLOCK_SIZE);

  // actual computation: Call the kernel
  cu_fillArray<<<dimgrid, dimblock>>>(array_d);

  // transfer results back to host
  result = cudaMemcpy(array, array_d, sizeof(int) * arraySize,
                      cudaMemcpyDeviceToHost);
  if (result != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed.");
    exit(1);
  }

  // release the memory on the GPU
  result = cudaFree(array_d);
  if (result != cudaSuccess) {
    fprintf(stderr, "cudaFree failed.");
    exit(1);
  }
}
