#pragma once
#include <exception>
#include <fmt/format.h>
#include <string>
#include <utility>

namespace ts {

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
class TensorException : std::exception {
  public:
    std::string _message;
    TensorException() : TensorException("cannot resize a non-owner") {}
    explicit TensorException(std::string message) : _message(std::move(message)) {}

    [[nodiscard]] const char *what() const noexcept override { return _message.data(); }
};

} // namespace ts