/*
 * compile both driver code and kernel code with nvcc, as in:
 * 			nvcc simple.c simple.cu
 */

#include <stdio.h>
#define SIZEOFARRAY 64

// The function fillArray is in the file simple.cu
extern void gpu_dotProduct(int *a, int size);

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
  // Declare the array and initialize to 0
  int a[SIZEOFARRAY];
  int i;
  for (i = 0; i < SIZEOFARRAY; i++) {
    a[i] = 0;
  }

  // Print the initial array
  printf("Initial state of the array:\n");
  for (i = 0; i < SIZEOFARRAY; i++) {
    printf("%d ", a[i]);
  }
  printf("\n");

  // Call the function that will call the GPU function
  fillArray(a, SIZEOFARRAY);

  // Again, print the array
  printf("Final state of the array:\n");
  for (i = 0; i < SIZEOFARRAY; i++) {
    printf("%d ", a[i]);
  }
  printf("\n");

  return 0;
}
