#include <catch2/catch.hpp>
#include <iostream>

using namespace std;

template<typename T>
void print_matrix(T** x) {
    for (std::size_t i = 0; i < 2; i++) {
        for (std::size_t j = 0; j < 3; j++) {
            std::cout << x[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

template<typename T>
void print_matrix(std::vector<std::vector<T>> x) {
    for (auto row : x) {
       for (T value: row) {
           std::cout << value << " ";
       }
       std::cout << std::endl;
    }
}

TEST_CASE("example of plain array") {
    // We cannot redefine it size and shapes cannot be values. They should be
    // constexpr/const
   float x[2][3] = {{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}};

   // Another way is using pointers
   auto **y = new float* [2];
   for (std::size_t i = 0; i < 2; i++) {
       y[i] = new float[3];
       for (std::size_t j = 0; j < 3; j++) {
          y[i][j] = 1.0;
       }
   }

   // Now we could check if both arrays hold the same values
   for (std::size_t i = 0; i < 2; i++) {
       for (std::size_t j = 0; j < 3; j++) {
          assert(x[i][j] == y[i][j]);
       }
   }
   print_matrix<float>(y);
}

TEST_CASE("use of std::vector") {
    std::vector<std::vector<float>> y;
    y.resize(2);
    for (std::size_t i = 0; i < 2; ++i) {
       y[i].resize(3);
       for (std::size_t j = 0; j < 3; ++j) {
          y[i][j] = 1.0;
       }
    }
    print_matrix<float>(y);
}