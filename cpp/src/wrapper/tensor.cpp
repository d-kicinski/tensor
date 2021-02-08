#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <tensor/tensor.hpp>
#include <tensor/nn/cross_entropy_loss.hpp>

namespace py = pybind11;

using size_type = ts::Tensor<float, 2>::size_type;

auto wrap_tensor2F(pybind11::module & m)
{
    py::class_<ts::Tensor<float, 2>>(m, "Tensor2F", py::buffer_protocol())
        .def(py::init<size_type , size_type>())

        // Construct from a buffer
        .def(py::init([](py::buffer const b) {
          py::buffer_info info = b.request();
          if (info.format != py::format_descriptor<float>::format() || info.ndim != 2)
              throw std::runtime_error("Incompatible buffer format!");

          auto v = new ts::Tensor<float, 2>(info.shape[0], info.shape[1]);
          memcpy(v->data()->data(),
                 info.ptr,
                 sizeof(float) * (size_t)(v->shape(0) * v->shape(1)));
          return v;
        }))

        .def("shape", [](ts::Tensor<float, 2> const  &t) -> std::array<int, 2> {
          return t.shape();
        })

        .def("data_size", &ts::Tensor<float, 2>::data_size)

        // Bare bones interface
        .def("__getitem__",
             [](ts::Tensor<float, 2> const &t, std::pair<py::ssize_t, py::ssize_t> i) {
               if (i.first >= t.shape(0) || i.second >= t.shape(1))
                   throw py::index_error();
               return t(i.first, i.second);
             })

        .def("__setitem__",
             [](ts::Tensor<float, 2> &t, std::pair<py::ssize_t, py::ssize_t> i, float v) {
               if (i.first >= t.shape(0) || i.second >= t.shape(1))
                   throw py::index_error();
               t(i.first, i.second) = v;
             })

        // Provide buffer access
        .def_buffer([](ts::Tensor<float, 2> &t) -> py::buffer_info {
          return py::buffer_info(
              t.data()->data(),                          /* Pointer to buffer */
              {t.shape(0), t.shape(1)},              /* Buffer dimensions */
              {sizeof(float) * size_t(t.shape(1)), /* Strides (in bytes) for each index */
               sizeof(float)});
        });
}

auto wrap_tensor1I(pybind11::module & m)
{

    py::class_<ts::Tensor<int, 1>>(m, "Tensor1I", py::buffer_protocol())
        .def(py::init<size_type>())

        // Construct from a buffer
        .def(py::init([](py::buffer const b) {
          py::buffer_info info = b.request();
          if (info.format != py::format_descriptor<int>::format() || info.ndim != 1)
              throw std::runtime_error("Incompatible buffer format!");

          auto v = new ts::Tensor<int, 1>(info.shape[0]);
          memcpy(v->data()->data(),
                 info.ptr,
                 sizeof(float) * (size_t)(v->shape(0)));
          return v;
        }))

        .def("shape", [](ts::Tensor<int, 1> const  &t) -> std::array<int, 1> {
          return t.shape();
        })

        .def("data_size", &ts::Tensor<int, 1>::data_size)

        // Bare bones interface
        .def("__getitem__",
             [](ts::Tensor<int, 1> const &t, py::ssize_t i) {
               if (i >= t.shape(0))
                   throw py::index_error();
               return t(i);
             })

        .def("__setitem__",
             [](ts::Tensor<int, 2> &t, py::ssize_t i, int v) {
               if (i >= t.shape(0))
                   throw py::index_error();
               t(i) = v;
             })

        // Provide buffer access
        .def_buffer([](ts::Tensor<int, 1> & t) -> py::buffer_info {
          return py::buffer_info(
                t.data()->data(),
                sizeof(int),
                py::format_descriptor<int>::format(),
                1,
                {t.shape(0)},
                {sizeof(int)}
          );
        });
}

auto wrap_ops(pybind11::module & m)
{
    m.def("dot",
          py::overload_cast<ts::Matrix const &, ts::Matrix const &, bool , bool>(&ts::dot),
          py::arg("A"), py::arg("B"), py::arg("A_T") = false, py::arg("B_T") = false);

    m.def("add", &ts::add<float, 2>);

    m.def("multiply", py::overload_cast<ts::Matrix const &, ts::Matrix const&>(&ts::multiply<float, 2>));

    m.def("log", &ts::log<float, 2>);

    m.def("pow", &ts::pow<float, 2>);

    m.def("exp", &ts::exp<float, 2>);

    m.def("transpose", &ts::transpose);

    m.def("sum", py::overload_cast<ts::Matrix const &, int>(&ts::sum_v2));

    m.def("get", [](ts::Matrix const & m, ts::Matrix const & i) {
      return ts::Matrix({ts::get(m, i[0].cast<int>())});
    });
}

auto wrap_nn(pybind11::module & m)
{
    py::class_<ts::CrossEntropyLoss>(m, "CrossEntropyLoss")
        .def(py::init<>())
        .def("__call__", &ts::CrossEntropyLoss::operator())
        .def("forward", &ts::CrossEntropyLoss::forward)
        .def("backward", &ts::CrossEntropyLoss::backward);
}

PYBIND11_MODULE(pytensor, m)
{
    wrap_tensor2F(m);
    wrap_tensor1I(m);
    wrap_ops(m);
    wrap_nn(m);
}
