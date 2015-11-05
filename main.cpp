/*
 * compile both driver code and kernel code with nvcc, as in:
 * 			nvcc simple.c simple.cu
 */
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>

// The function fillArray is in the file simple.cu
extern "C" void gpu_dotProduct(int *result_array, int num_vectors);

void gen_force_array(int *force, int num_vectors) {
  int i, j, half_vectors;
  half_vectors = num_vectors / 2;
  // go up
  for (i = 0; i < half_vectors; i++) {
    force[i] = i + 1;
  }

  // walk backwards with array, forward counting
  i = 1;
  for (j = num_vectors; j > half_vectors; j--) {
    force[j - 1] = i;
    i++;
  }

  return;
}

void gen_distance_array(int *distance, int num_vectors) {
  int i = 0, j = 1;
  for (i = 0; i < num_vectors; i++) {
    distance[i] = j;
    j++;
    if (j > 10) {
      j = 1;
    }
  }
  return;
}

int main(int argc, char *argv[]) {
  int num_vectors = 0, half_vectors, i, j, s_dotproduct = 0,
      pu_dotproduct_result = 0, temp;
  if (argc < 2) {
    std::cout << "Usage: cuda-musclework num_vectors" << std::endl;
  }
  num_vectors = atoi(argv[1]);
  std::cout << "Using num_vectors: " << num_vectors << std::endl;
  int *result = (int *)malloc(num_vectors * sizeof(int));

  // Call the function that will call the GPU function
  clock_t gpu_start = clock();
  gpu_dotProduct(result, num_vectors);
  clock_t gpu_stop = clock();
  double elapsed_gpu = double(gpu_stop - gpu_start) / (CLOCKS_PER_SEC / 1000);

  for (i = 0; i < num_vectors; i++) {
    pu_dotproduct_result = pu_dotproduct_result + result[i];
  }

  std::cout << "GPU Calc Total: " << pu_dotproduct_result << std::endl;
  std::cout << "GPU Time Taken (msec): " << elapsed_gpu << std::endl;

  return 0;
}
