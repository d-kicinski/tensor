#pragma once
#include <exception>
#include <string>
#include <fmt/format.h>

namespace space {

class InvalidShapeException: std::exception {
   [[nodiscard]] const char * what() const noexcept override {
       return "Invalid shapes";
   }
};

class IndexOutOfRangeException: std::exception {
  public:
    IndexOutOfRangeException(int index) {
        _message = fmt::format("Index {} out of range.", _index);
    }
  private:
    int _index;
    std::string _message;

    [[nodiscard]] const char * what() const noexcept override {
        return _message.data();
    }
};
}