#pragma once
#include <pybind11/pybind11.h>
#include <tensor/nn/grad_holder.hpp>

class PyGradHolderFloat : public ts::GradHolder<float>{
  public:
    using ts::GradHolder<float>::GradHolder;

    auto grad() -> DataHolderRef override {
        PYBIND11_OVERRIDE_PURE(
            DataHolderRef, /* Return type */
            ts::GradHolder<float>,      /* Parent class */
            grad          /* Name of function in C++ (must match Python name) */
        );
    }

    auto tensor() -> DataHolderRef override {
        PYBIND11_OVERRIDE_PURE(
            DataHolderRef, /* Return type */
            ts::GradHolder<float>,      /* Parent class */
            tensor          /* Name of function in C++ (must match Python name) */
        );
    }

//    auto name() -> DataHolderRef override {
//        PYBIND11_OVERRIDE(
//            DataHolderRef, /* Return type */
//            ts::GradHolder<float>,      /* Parent class */
//            name          /* Name of function in C++ (must match Python name) */
//        );
//    }
};

class PyGradHolderInt : public ts::GradHolder<int>{
  public:
    using ts::GradHolder<int>::GradHolder;

    auto grad() -> DataHolderRef override {
        PYBIND11_OVERRIDE_PURE(
            DataHolderRef, /* Return type */
            ts::GradHolder<int>,      /* Parent class */
            grad          /* Name of function in C++ (must match Python name) */
        );
    }

    auto tensor() -> DataHolderRef override {
        PYBIND11_OVERRIDE_PURE(
            DataHolderRef, /* Return type */
            ts::GradHolder<int>,      /* Parent class */
            tensor          /* Name of function in C++ (must match Python name) */
        );
    }

//    auto name() -> DataHolderRef override {
//        PYBIND11_OVERRIDE(
//            DataHolderRef, /* Return type */
//            ts::GradHolder<int>,      /* Parent class */
//            name          /* Name of function in C++ (must match Python name) */
//        );
//    }
};
