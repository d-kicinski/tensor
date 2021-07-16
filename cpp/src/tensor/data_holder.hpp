#pragma once

#include <memory>
#include <vector>

namespace ts {

template <typename Element> class DataHolder {
  public:
    using vector_t = std::vector<Element>;
    using data_ptr_t = std::shared_ptr<vector_t>;
    using iterator = typename vector_t::iterator;

    virtual auto get() const -> data_ptr_t = 0;
    virtual auto begin() const -> iterator = 0;
    virtual auto end() const -> iterator = 0;
};

} // namespace ts
