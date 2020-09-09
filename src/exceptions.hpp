#pragma once
#include <exception>
#include <fmt/format.h>
#include <string>
#include <utility>

namespace space {

class InvalidShapeException : std::exception {
    [[nodiscard]] const char *what() const noexcept override { return "Invalid shapes"; }
};

class IndexOutOfRangeException : std::exception {
  public:
    IndexOutOfRangeException(int index)
    {
        _message = fmt::format("Index {} out of range.", _index);
    }

    [[nodiscard]] const char *what() const noexcept override { return _message.data(); }

  private:
    int _index;
    std::string _message;
};
class FlatArrayException : std::exception {
  public:
    std::string _message;
    FlatArrayException() : FlatArrayException("cannot resize a non-owner") {}
    FlatArrayException(std::string message) : _message(std::move(message)) {}

    [[nodiscard]] const char *what() const noexcept override { return _message.data(); }
};
} // namespace space