clang-format -i main_cpu.cpp
clang-format -i main.cpp
clang-format -i main.cu
clang++ -std=c++14 -O2 main_cpu.cpp -o cuda-musclework
chmod +x cuda-musclework
