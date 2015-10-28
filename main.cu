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
__global__ void cu_fillArray(int *block_d, int *thread_d) {
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
