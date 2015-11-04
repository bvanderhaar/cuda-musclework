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
__global__ void cu_dotProduct(int *distance_array_d, int *force_array_d,
                              int *result_array_d, int max) {
  int x;
  /* blockIdx.x is a built-in variable in CUDA
     that returns the blockId in the x axis.
     threadIdx.x is another built-in variable in CUDA
     that returns the threadId in the x axis
     of the thread that is being executed by the
     stream processor accessing this particular block
  */
  x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (x < max) {
    result_array_d[x] = distance_array_d[x] * force_array_d[x];
  }
}

// Called from driver program.  Handles running GPU calculation
extern "C" void gpu_dotProduct(int *distance_array, int *force_array,
                               int *result_array int num_vectors) {
  int *distance_array_d;
  int *force_array_d;
  int *result_array_d;

  // allocate space in the device
  cudaMalloc((void **)&distance_array_d, sizeof(int) * num_vectors);
  cudaMalloc((void **)&force_array_d, sizeof(int) * num_vectors);
  cudaMalloc((void **)&result_array_d, sizeof(int) * num_vectors);

  // copy the array from host to array_d in the device
  cudaMemcpy(distance_array_d, distance_array, sizeof(int) * num_vectors,
             cudaMemcpyHostToDevice);

  cudaMemcpy(force_array_d, force_array, sizeof(int) * num_vectors,
             cudaMemcpyHostToDevice);

  cudaMemcpy(result_array_d, result_array, sizeof(int) * num_vectors,
             cudaMemcpyHostToDevice);

  // set execution configuration
  dim3 dimblock(BLOCK_SIZE);
  dim3 dimgrid(ceil((double)num_vectors / BLOCK_SIZE));

  // actual computation: Call the kernel
  cu_dotProduct<<<dimgrid, dimblock>>>(distance_array_d, force_array_d,
                                       result_array_d, num_vectors);

  // transfer results back to host
  cudaMemcpy(result_array, result_array_d, sizeof(int) * num_vectors,
             cudaMemcpyDeviceToHost);

  // release the memory on the GPU
  cudaFree(distance_array_d);
  cudaFree(force_array_d);
  cudaFree(result_array_d);
}
