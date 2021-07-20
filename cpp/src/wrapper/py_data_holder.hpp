#pragma once
#include <pybind11/pybind11.h>
#include <tensor/data_holder.hpp>

class PyDataHolderFloat : public ts::DataHolder<float>{
  public:
    using Element = float;
    using vector_t = std::vector<Element>;
    using data_ptr_t = std::shared_ptr<vector_t>;
    using iterator = typename vector_t::iterator;

    using ts::DataHolder<float>::DataHolder;

    float const dummy = 0; // hack for at() method

    auto get() const -> data_ptr_t override { return ts::DataHolder<float>::data_ptr_t(); }
    auto begin() const -> iterator override { return ts::DataHolder<float>::iterator(); }
    auto end() const -> iterator override { return ts::DataHolder<float>::iterator(); }
    auto at(ts::size_type i) const -> Element & override { return const_cast<Element &>(dummy); }
};


class PyDataHolderInt : public ts::DataHolder<int>{
  public:
    using Element = int;
    using vector_t = std::vector<Element>;
    using data_ptr_t = std::shared_ptr<vector_t>;
    using iterator = typename vector_t::iterator;

    using ts::DataHolder<int>::DataHolder;

    int const dummy = 0; // hack for at() method

    auto get() const -> data_ptr_t override { return ts::DataHolder<int>::data_ptr_t(); }
    auto begin() const -> iterator override { return ts::DataHolder<int>::iterator(); }
    auto end() const -> iterator override { return ts::DataHolder<int>::iterator(); }
    auto at(ts::size_type i) const -> Element & override { return const_cast<Element &>(dummy); }
};
