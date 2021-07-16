#pragma once

#include <memory>
#include <string>

#include "tensor/data_holder.hpp"

namespace ts {

template <typename Element> class GradHolder {
  public:
    using DataHolderRef = DataHolder<Element> &;

    virtual auto grad() -> DataHolderRef = 0;

    virtual auto tensor() -> DataHolderRef = 0;

    virtual auto name() -> std::string { return _name; };

  private:
    std::string _name = "GradHolder";
};

} // namespace ts