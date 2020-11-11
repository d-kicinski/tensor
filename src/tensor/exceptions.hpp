#pragma once
#include <exception>
#include <string>
#include <utility>

namespace ts {

class TensorException : std::exception {
  public:
    std::string _message;
    TensorException() : TensorException("cannot resize a non-owner") {}
    explicit TensorException(std::string message) : _message(std::move(message)) {}

    [[nodiscard]] const char *what() const noexcept override { return _message.data(); }
};

} // namespace ts