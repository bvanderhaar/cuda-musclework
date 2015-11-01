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
extern "C" void gpu_dotProduct(int *distance_array, int *force_array,
                               int num_vectors) {
  // a_d is the GPU counterpart of the array that exists in host memory
  int *distance_array_d;
  int *force_array_d;
  int result_array[num_vertices];
  int *result_array_d;
  cudaError_t result;

  // allocate space in the device
  result = cudaMalloc((void **)&distance_array_d, sizeof(int) * num_vertices);
  if (result != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed.");
    exit(1);
  }

  result = cudaMalloc((void **)&force_array_d, sizeof(int) * num_vertices);
  if (result != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed.");
    exit(1);
  }

  result = cudaMalloc((void **)&result_array_d, sizeof(int) * num_vertices);
  if (result != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed.");
    exit(1);
  }

  // copy the array from host to array_d in the device
  result = cudaMemcpy(distance_array_d, distance_array,
                      sizeof(int) * num_vertices, cudaMemcpyHostToDevice);
  if (result != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed.");
    exit(1);
  }

  result = cudaMemcpy(force_array_d, force_array, sizeof(int) * num_vertices,
                      cudaMemcpyHostToDevice);
  if (result != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed.");
    exit(1);
  }

  result = cudaMemcpy(result_array_d, result_array, sizeof(int) * num_vertices,
                      cudaMemcpyHostToDevice);
  if (result != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed.");
    exit(1);
  }

  // set execution configuration
  dim3 dimblock(BLOCK_SIZE);
  dim3 dimgrid(num_vertices / BLOCK_SIZE);

  // actual computation: Call the kernel
  cu_dotProduct<<<dimgrid, dimblock>>>(result_array_d);

  // transfer results back to host
  result = cudaMemcpy(result_array, result_array_d, sizeof(int) * arraySize,
                      cudaMemcpyDeviceToHost);
  if (result != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed.");
    exit(1);
  }

  // release the memory on the GPU
  result = cudaFree(distance_array_d);
  if (result != cudaSuccess) {
    fprintf(stderr, "cudaFree failed.");
    exit(1);
  }

  // release the memory on the GPU
  result = cudaFree(force_array_d);
  if (result != cudaSuccess) {
    fprintf(stderr, "cudaFree failed.");
    exit(1);
  }

  // release the memory on the GPU
  result = cudaFree(result_array_d);
  if (result != cudaSuccess) {
    fprintf(stderr, "cudaFree failed.");
    exit(1);
  }
}
