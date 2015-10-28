clang-format -i main.cpp
clang-format -i main.cu
clang++ -std=c++14 -O2 main.cpp -o cuda-musclework
chmod +x cuda-musclework
