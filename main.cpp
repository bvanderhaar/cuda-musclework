/*
 * compile both driver code and kernel code with nvcc, as in:
 * 			nvcc simple.c simple.cu
 */

#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>

// The function fillArray is in the file simple.cu
extern void gpu_dotProduct(int *distance_array, int *force_array,
                           int num_vectors);

std::vector<int> gen_force_array(int num_vectors) {
  int i, j, half_vectors;
  half_vectors = num_vectors / 2;
  std::vector<int> force_array(num_vectors);

  // go up
  for (i = 0; i < half_vectors; i++) {
    force_array[i] = i + 1;
  }

  // walk backwards with array, forward counting
  i = 1;
  for (j = num_vectors; j > half_vectors; j--) {
    force_array[j - 1] = i;
    i++;
  }

  return force_array;
}

std::vector<int> gen_distance_array(int num_vectors) {
  int i = 0, j = 1;
  std::vector<int> distance_array(num_vectors);
  for (i = 0; i < num_vectors; i++) {
    distance_array[i] = j;
    j++;
    if (j > 10) {
      j = 1;
    }
  }
  return distance_array;
}

int main(int argc, char *argv[]) {
  int num_vectors = 0, half_vectors, i, j, s_dotproduct = 0,
      pu_dotproduct_result = 0, temp;
  if (argc < 2) {
    std::cout << "Usage: cuda-musclework num_vectors" << std::endl;
  }
  num_vectors = atoi(argv[1]);
  std::cout << "Using num_vectors: " << num_vectors << std::endl;
  std::vector<int> distance;
  std::vector<int> force;

  distance.resize(num_vectors, 0);
  force.resize(num_vectors, 0);

  distance = gen_distance_array(num_vectors);
  force = gen_force_array(num_vectors);

  // Call the function that will call the GPU function
  gpu_dotProduct(distance, force, num_vectors);

  return 0;
}
