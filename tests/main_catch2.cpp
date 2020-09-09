#define CATCH_CONFIG_MAIN // provides main(); this line is required in only one .cpp file
#include <catch2/catch.hpp>

#include <iostream>

class Foo {
  public:
    int value = 0;
    Foo(){
        value = 10;
    };
};

template <typename Object>
class Array {
  public:
    Object* data;

    Array(size_t size) {
       data = new Object[size];
    }
};

TEST_CASE("just sanity check") {
   auto * arr = new Foo[3];
   Array<Foo> arr_foo(3);

   for (int i = 0; i < 3; i++) {
       std::cout << arr[i].value << std::endl;
   }

    for (int i = 0; i < 3; i++) {
        std::cout << arr_foo.data[i].value << std::endl;
    }
}