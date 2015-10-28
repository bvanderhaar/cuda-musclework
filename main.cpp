#include <stdio.h>
#include <string>
#include <iostream>

int main(int argc, char *argv[]) {
  int num_vectors = 0, force[], distance[], i, j;
  if (argc < 2) {
    std::cout << "Usage: cuda-musclework num_vectors" << std::endl;
  }

  num_vectors = atoi(argv[1]);

  // go up
  for (i = 0; i < num_vectors; i++) {
    force[i] = i;
  }

  // back down
  for (j = num_vectors; j < 0 ; j--) {
    force[j] = i - j;
  }

}
