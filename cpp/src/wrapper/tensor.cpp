#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <tensor/tensor.hpp>
#include <tensor/nn/cross_entropy_loss.hpp>

namespace py = pybind11;

using size_type = ts::Tensor<float, 2>::size_type;

template <typename Element>
auto wrap_tensor2D(pybind11::module & m, char const * class_name)
{
    py::class_<ts::Tensor<Element, 2>>(m, class_name, py::buffer_protocol())
        .def(py::init<size_type , size_type>())

        // Construct from a buffer
        .def(py::init([](py::buffer const b) {
          py::buffer_info info = b.request();
          if (info.format != py::format_descriptor<Element>::format() || info.ndim != 2)
              throw std::runtime_error("Incompatible buffer format!");

          auto v = new ts::Tensor<Element, 2>(info.shape[0], info.shape[1]);
          memcpy(v->data()->data(),
                 info.ptr,
                 sizeof(Element) * (size_t)(v->shape(0) * v->shape(1)));
          return v;
        }))

        .def("shape", [](ts::Tensor<Element, 2> const  &t) -> std::array<int, 2> {
          return t.shape();
        })

        .def("data_size", &ts::Tensor<Element, 2>::data_size)

        // Bare bones interface
        .def("__getitem__",
             [](ts::Tensor<Element, 2> const &t, std::pair<py::ssize_t, py::ssize_t> i) {
               if (i.first >= t.shape(0) || i.second >= t.shape(1))
                   throw py::index_error();
               return t(i.first, i.second);
             })

        .def("__setitem__",
             [](ts::Tensor<Element, 2> &t, std::pair<py::ssize_t, py::ssize_t> i, Element v) {
               if (i.first >= t.shape(0) || i.second >= t.shape(1))
                   throw py::index_error();
               t(i.first, i.second) = v;
             })

        // Provide buffer access
        .def_buffer([](ts::Tensor<Element, 2> &t) -> py::buffer_info {
          return py::buffer_info(
              t.data()->data(),                          /* Pointer to buffer */
              {t.shape(0), t.shape(1)},              /* Buffer dimensions */
              {sizeof(Element) * size_t(t.shape(1)), /* Strides (in bytes) for each index */
               sizeof(Element)});
        });
}

template <typename Element>
auto wrap_tensor1D(pybind11::module & m, char const * class_name)
{

    py::class_<ts::Tensor<Element, 1>>(m, class_name, py::buffer_protocol())
        .def(py::init<size_type>())

        // Construct from a buffer
        .def(py::init([](py::buffer const b) {
          py::buffer_info info = b.request();
          if (info.format != py::format_descriptor<Element>::format() || info.ndim != 1)
              throw std::runtime_error("Incompatible buffer format!");

          auto v = new ts::Tensor<Element, 1>(info.shape[0]);
          memcpy(v->data()->data(),
                 info.ptr,
                 sizeof(Element) * (size_t)(v->shape(0)));
          return v;
        }))

        .def("shape", [](ts::Tensor<Element, 1> const  &t) -> std::array<int, 1> {
          return t.shape();
        })

        .def("data_size", &ts::Tensor<Element, 1>::data_size)

        // Bare bones interface
        .def("__getitem__",
             [](ts::Tensor<Element, 1> const &t, py::ssize_t i) {
               if (i >= t.shape(0))
                   throw py::index_error();
               return t(i);
             })

        .def("__setitem__",
             [](ts::Tensor<Element, 1> &t, py::ssize_t i, int v) {
               if (i >= t.shape(0))
                   throw py::index_error();
               t(i) = v;
             })

        // Provide buffer access
        .def_buffer([](ts::Tensor<Element, 1> & t) -> py::buffer_info {
          return py::buffer_info(
                t.data()->data(),
                sizeof(Element),
                py::format_descriptor<Element>::format(),
                1,
                {t.shape(0)},
                {sizeof(Element)}
          );
        });
}

auto wrap_ops(pybind11::module & m)
{
    m.def("dot",
          py::overload_cast<ts::MatrixF const &, ts::MatrixF const &, bool , bool>(&ts::dot),
          py::arg("A"), py::arg("B"), py::arg("A_T") = false, py::arg("B_T") = false);

    m.def("add_matrixf_matrixf", &ts::add<float, 2>);
    m.def("add_matrixi_matrixi", &ts::add<int, 2>);
    m.def("add_vectorf_vectorf", &ts::add<float, 1>);
    m.def("add_vectori_vectori", &ts::add<int, 1>);
    m.def("add_matrixf_vectorf", &ts::add<float>);
    m.def("add_matrixi_vectori", &ts::add<int>);

    m.def("multiply_vectorf_vectorf", py::overload_cast<ts::VectorF const &, ts::VectorF const&>(&ts::multiply<float, 1>));
    m.def("multiply_matrixf_matrixf", py::overload_cast<ts::MatrixF const &, ts::MatrixF const&>(&ts::multiply<float, 2>));
    m.def("multiply_vectorf_f", py::overload_cast<ts::VectorF const &, float>(&ts::multiply<float, 1>));
    m.def("multiply_matrixf_f", py::overload_cast<ts::MatrixF const &, float>(&ts::multiply<float, 2>));

    m.def("log", &ts::log<float, 2>);

    m.def("pow", &ts::pow<float, 2>);

    m.def("exp", &ts::exp<float, 2>);

    m.def("transpose", &ts::transpose);

    m.def("sum", py::overload_cast<ts::MatrixF const &, int>(&ts::sum_v2));

    m.def("get", [](ts::MatrixF const & m, ts::MatrixF const & i) {
      return ts::MatrixF({ts::get(m, i[0].cast<int>())});
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

PYBIND11_MODULE(libtensor, m)
{
    wrap_tensor2D<int>(m, "MatrixI");
    wrap_tensor2D<float>(m, "MatrixF");
    wrap_tensor1D<int>(m, "VectorI");
    wrap_tensor1D<float>(m, "VectorF");
    wrap_ops(m);
    wrap_nn(m);
}
