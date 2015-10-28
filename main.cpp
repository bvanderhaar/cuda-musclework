#include <stdio.h>
#include <string>
#include <iostream>

int main(int argc, char *argv[]) {
  int vertices = 0, i = 0, j = 0, k = 0;
  if (argc < 2) {
    std::cout << "Usage: floyds-algorithm num_vertices [-t]" << std::endl;
  }

  vertices = atoi(argv[1]);
}
