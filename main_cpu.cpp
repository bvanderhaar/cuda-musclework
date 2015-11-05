#include <stdio.h>
#include <string>
#include <iostream>
#include <vector>

std::vector<int> GDistance;
std::vector<int> GForce;

std::vector<int> gen_distance_array(int num_vectors) {
  int i = 0;
  std::vector<int> distance_array(num_vectors);
  for (i = 0; i < num_vectors; i++) {
    distance_array[i] = (i + 1) % 10;
    if (distance_array[i] == 0) {
      distance_array[i] = 10;
    }
  }
  return distance_array;
}

std::vector<int> gen_force_array(int num_vectors) {
  int i, j, half_vectors;
  half_vectors = num_vectors / 2;
  std::vector<int> force_array(num_vectors);

  for (i = 0; i < num_vectors; i++) {
    if (i < half_vectors) {
      force_array[i] = i + 1;
    } else {
      force_array[i] = half_vectors + (half_vectors - i);
    }
  }

  return force_array;
}

void printarray(std::vector<int> array, int size) {
  for (int i = 0; i < size; i++) {
    std::cout << array[i] << " ";
  }
}

int main(int argc, char *argv[]) {
  int num_vectors = 0, half_vectors, i, j, temp;
  long long s_dotproduct = 0;
  if (argc < 2) {
    std::cout << "Usage: cuda-musclework num_vectors" << std::endl;
  }
  num_vectors = atoi(argv[1]);
  std::cout << "Using num_vectors: " << num_vectors << std::endl;

  // generate and compute the dot product, serially
  clock_t cpu_start = clock();
  GDistance.resize(num_vectors, 0);
  GForce.resize(num_vectors, 0);
  GDistance = gen_distance_array(num_vectors);
  GForce = gen_force_array(num_vectors);
  for (i = 0; i < num_vectors; i++) {
    temp = GDistance[i] * GForce[i];
    s_dotproduct = s_dotproduct + temp;
  }
  clock_t cpu_stop = clock();
  double elapsed_cpu = double(cpu_stop - cpu_start) / (CLOCKS_PER_SEC / 1000);

  // print array
  // std::cout << "Force Array: \t\t";
  // printarray(GForce, num_vectors);
  // std::cout << std::endl << "Distance Array: \t";
  // printarray(GDistance, num_vectors);
  // std::cout << std::endl;
  std::cout << "CPU (serial) dot product: " << s_dotproduct << std::endl;
  // std::cout << "Parallel dot product: " << pu_dotproduct_result << std::endl;
  std::cout << "CPU Time Taken (msec): " << elapsed_cpu << std::endl;
}
