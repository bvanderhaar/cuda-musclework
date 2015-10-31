#include <stdio.h>
#include <string>
#include <iostream>
#include <thread>
#include <vector>

const int NUM_THREADS = 16;

std::vector<int> GResultArray(NUM_THREADS);
std::vector<int> GDistance;
std::vector<int> GForce;

void pu_dotproduct(int start, int end, int id) {
  int temp = 0, i, product = 0;
  for (i = start; i < end; i++) {
    temp = GDistance[i] * GForce[i];
    product = product + temp;
  }
  GResultArray[id] = product;
  return;
}

void printarray(std::vector<int> array, int size) {
  for (int i = 0; i < size; i++) {
    std::cout << array[i] << " ";
  }
}

int main(int argc, char *argv[]) {
  int num_vectors = 0, half_vectors, i, j, s_dotproduct = 0,
      pu_dotproduct_result = 0, temp;
  if (argc < 2) {
    std::cout << "Usage: cuda-musclework num_vectors" << std::endl;
  }
  num_vectors = atoi(argv[1]);
  half_vectors = num_vectors / 2;
  std::cout << "Using num_vectors: " << num_vectors << std::endl;
  std::cout << "half_vectors: " << half_vectors << std::endl;
  GDistance.resize(num_vectors, 0);
  GForce.resize(num_vectors, 0);

  // go up
  for (i = 0; i < half_vectors; i++) {
    GForce[i] = i + 1;
  }

  // walk backwards with array, forward counting
  i = 1;
  for (j = num_vectors; j > half_vectors; j--) {
    GForce[j - 1] = i;
    i++;
  }

  j = 1;
  for (i = 0; i < num_vectors; i++) {
    GDistance[i] = j;
    j++;
    if (j > 10) {
      j = 1;
    }
  }

  // compute the dot product, serially
  for (i = 0; i < num_vectors; i++) {
    temp = GDistance[i] * GForce[i];
    s_dotproduct = s_dotproduct + temp;
  }

  // compute the dot product with threads
  int start, end, id, work_thread_size;
  work_thread_size = num_vectors / NUM_THREADS - 1;
  std::cout << "Work thread size: " << work_thread_size << std::endl;

  std::vector<std::thread> t(NUM_THREADS);
  for (id = 0; id < NUM_THREADS; id++) {
    start = id * work_thread_size;
    end = start + work_thread_size;
    if (id == (NUM_THREADS - 1)) {
      end = num_vectors;
    }
    std::cout << "Thread id: " << id << " start: " << start << " end: " << end
              << std::endl;
    t[id] = std::thread(pu_dotproduct, start, end, id);
  }

  for (i = 0; i < NUM_THREADS; i++) {
    std::cout << "Joining thread id: " << i << std::endl;
    t[i].join();
  }

  std::cout << "Threaded result array: ";
  printarray(GResultArray, NUM_THREADS);
  std::cout << std::endl;
  for (i = 0; i < NUM_THREADS; i++) {
    pu_dotproduct_result = pu_dotproduct_result + GResultArray[i];
  }

  // print array
  std::cout << "Force Array: \t\t";
  printarray(GForce, num_vectors);
  std::cout << std::endl << "Distance Array: \t";
  printarray(GDistance, num_vectors);
  std::cout << std::endl;
  std::cout << "Serial dot product: " << s_dotproduct << std::endl;
  std::cout << "Parallel dot product: " << pu_dotproduct_result << std::endl;
}
