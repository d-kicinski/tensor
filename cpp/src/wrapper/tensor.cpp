#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <tensor/nn/activations.hpp>
#include <tensor/nn/cross_entropy_loss.hpp>
#include <tensor/nn/layer/conv_2d.hpp>
#include <tensor/nn/layer/feed_forward.hpp>
#include <tensor/nn/layer/max_pool_2d.hpp>
#include <tensor/nn/max_pool_2d.hpp>
#include <tensor/nn/optimizer/adagrad.hpp>
#include <tensor/nn/optimizer/sgd.hpp>
#include <tensor/nn/softmax.hpp>
#include <tensor/tensor.hpp>

#include "py_data_holder.hpp"
#include "py_grad_holder.hpp"

#ifdef TENSOR_USE_PROTOBUF
#include <tensor/nn/saver.hpp>
#endif

namespace py = pybind11;

using size_type = ts::size_type;

template <typename Element> auto wrap_tensor4D(pybind11::module &m, char const *class_name)
{
    py::class_<ts::Tensor<Element, 4>, ts::DataHolder<Element>>(m, class_name, py::buffer_protocol())
        .def(py::init<size_type, size_type, size_type, size_type>())

        // Construct from a buffer
        .def(py::init([](py::buffer const b) {
            py::buffer_info info = b.request();
            if (info.format != py::format_descriptor<Element>::format() || info.ndim != 4)
                throw std::runtime_error("Incompatible buffer format!");

            auto v = new ts::Tensor<Element, 4>(info.shape[0], info.shape[1], info.shape[2], info.shape[3]);
            memcpy(v->data()->data(), info.ptr,
                   sizeof(Element) * (size_t)(v->shape(0) * v->shape(1) * v->shape(2) * v->shape(3)));
            return v;
        }))

        .def("shape", [](ts::Tensor<Element, 4> const &t) -> std::array<size_type, 4> { return t.shape(); })

        .def("reshape4", &ts::Tensor<Element, 4>::template reshape<4>)
        .def("reshape3", &ts::Tensor<Element, 4>::template reshape<3>)
        .def("reshape2", &ts::Tensor<Element, 4>::template reshape<2>)
        .def("reshape1", &ts::Tensor<Element, 4>::template reshape<1>)

        .def("data_size", &ts::Tensor<Element, 4>::data_size)

        // Bare bones interface
        .def("__getitem__",
             [](ts::Tensor<Element, 4> const &t, std::tuple<py::ssize_t, py::ssize_t, py::ssize_t, py::size_t> i) {
                 auto [dim_0, dim_1, dim_2, dim_3] = i;
                 if (dim_0 >= t.shape(0) || dim_1 >= t.shape(1) || dim_2 >= t.shape(2) || dim_3 >= t.shape(3))
                     throw py::index_error();
                 return t(dim_0, dim_1, dim_2, dim_3);
             })

        .def("__setitem__",
             [](ts::Tensor<Element, 4> &t, std::tuple<py::ssize_t, py::ssize_t, py::ssize_t, py::size_t> i, Element v) {
                 auto [dim_0, dim_1, dim_2, dim_3] = i;
                 if (dim_0 >= t.shape(0) || dim_1 >= t.shape(1) || dim_2 >= t.shape(2) || dim_3 >= t.shape(3))
                     throw py::index_error();
                 t(dim_0, dim_1, dim_2, dim_3) = v;
             })

        // Provide buffer access
        .def_buffer([](ts::Tensor<Element, 4> &t) -> py::buffer_info {
            return py::buffer_info(t.data()->data(),                                 // Pointer to buffer
                                   {t.shape(0), t.shape(1), t.shape(2), t.shape(3)}, // Buffer dimensions
                                   {sizeof(Element) * size_t(t.shape(1)) * size_t(t.shape(2)) *
                                        size_t(t.shape(3)), // Strides (in bytes) for each index
                                    sizeof(Element) * size_t(t.shape(2)) * size_t(t.shape(3)),
                                    sizeof(Element) * size_t(t.shape(3)), sizeof(Element)});
        });
}

template <typename Element> auto wrap_tensor3D(pybind11::module &m, char const *class_name)
{
    py::class_<ts::Tensor<Element, 3>, ts::DataHolder<Element>>(m, class_name, py::buffer_protocol())
        .def(py::init<size_type, size_type, size_type>())

        // Construct from a buffer
        .def(py::init([](py::buffer const b) {
            py::buffer_info info = b.request();
            if (info.format != py::format_descriptor<Element>::format() || info.ndim != 3)
                throw std::runtime_error("Incompatible buffer format!");

            auto v = new ts::Tensor<Element, 3>(info.shape[0], info.shape[1], info.shape[2]);
            memcpy(v->data()->data(), info.ptr, sizeof(Element) * (size_t)(v->shape(0) * v->shape(1) * v->shape(2)));
            return v;
        }))

        .def("shape", [](ts::Tensor<Element, 3> const &t) -> std::array<size_type, 3> { return t.shape(); })

        .def("reshape4", &ts::Tensor<Element, 3>::template reshape<4>)
        .def("reshape3", &ts::Tensor<Element, 3>::template reshape<3>)
        .def("reshape2", &ts::Tensor<Element, 3>::template reshape<2>)
        .def("reshape1", &ts::Tensor<Element, 3>::template reshape<1>)

        .def("data_size", &ts::Tensor<Element, 3>::data_size)

        // Bare bones interface
        .def("__getitem__",
             [](ts::Tensor<Element, 3> const &t, std::tuple<py::ssize_t, py::ssize_t, py::ssize_t> i) {
                 auto [dim_0, dim_1, dim_2] = i;
                 if (dim_0 >= t.shape(0) || dim_1 >= t.shape(1) || dim_2 >= t.shape(2))
                     throw py::index_error();
                 return t(dim_0, dim_1, dim_2);
             })

        .def("__setitem__",
             [](ts::Tensor<Element, 3> &t, std::tuple<py::ssize_t, py::ssize_t, py::ssize_t> i, Element v) {
                 auto [dim_0, dim_1, dim_2] = i;
                 if (dim_0 >= t.shape(0) || dim_1 >= t.shape(1) || dim_2 >= t.shape(2))
                     throw py::index_error();
                 t(dim_0, dim_1, dim_2) = v;
             })

        // Provide buffer access
        .def_buffer([](ts::Tensor<Element, 3> &t) -> py::buffer_info {
            return py::buffer_info(
                t.data()->data(),                                           // Pointer to buffer
                {t.shape(0), t.shape(1), t.shape(2)},                       // Buffer dimensions
                {sizeof(Element) * size_t(t.shape(1)) * size_t(t.shape(2)), // Strides (in bytes) for each index
                 sizeof(Element) * size_t(t.shape(2)), sizeof(Element)});
        });
}

template <typename Element> auto wrap_tensor2D(pybind11::module &m, char const *class_name)
{
    py::class_<ts::Tensor<Element, 2>, ts::DataHolder<Element>>(m, class_name, py::buffer_protocol())
        .def(py::init<size_type, size_type>())

        // Construct from a buffer
        .def(py::init([](py::buffer const b) {
            py::buffer_info info = b.request();
            if (info.format != py::format_descriptor<Element>::format() || info.ndim != 2)
                throw std::runtime_error("Incompatible buffer format!");

            auto v = new ts::Tensor<Element, 2>(info.shape[0], info.shape[1]);
            memcpy(v->data()->data(), info.ptr, sizeof(Element) * (size_t)(v->shape(0) * v->shape(1)));
            return v;
        }))

        .def("shape", [](ts::Tensor<Element, 2> const &t) -> std::array<size_type, 2> { return t.shape(); })

        .def("reshape4", &ts::Tensor<Element, 2>::template reshape<4>)
        .def("reshape3", &ts::Tensor<Element, 2>::template reshape<3>)
        .def("reshape2", &ts::Tensor<Element, 2>::template reshape<2>)
        .def("reshape1", &ts::Tensor<Element, 2>::template reshape<1>)

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
            return py::buffer_info(t.data()->data(),                      /* Pointer to buffer */
                                   {t.shape(0), t.shape(1)},              /* Buffer dimensions */
                                   {sizeof(Element) * size_t(t.shape(1)), /* Strides (in bytes) for each index */
                                    sizeof(Element)});
        });
}

template <typename Element> auto wrap_tensor1D(pybind11::module &m, char const *class_name)
{

    py::class_<ts::Tensor<Element, 1>, ts::DataHolder<Element>>(m, class_name, py::buffer_protocol())
        .def(py::init<size_type>())

        // Construct from a buffer
        .def(py::init([](py::buffer const b) {
            py::buffer_info info = b.request();
            if (info.format != py::format_descriptor<Element>::format() || info.ndim != 1)
                throw std::runtime_error("Incompatible buffer format!");

            auto v = new ts::Tensor<Element, 1>(info.shape[0]);
            memcpy(v->data()->data(), info.ptr, sizeof(Element) * (size_t)(v->shape(0)));
            return v;
        }))

        .def("shape", [](ts::Tensor<Element, 1> const &t) -> std::array<size_type, 1> { return t.shape(); })

        .def("reshape4", &ts::Tensor<Element, 1>::template reshape<4>)
        .def("reshape3", &ts::Tensor<Element, 1>::template reshape<3>)
        .def("reshape2", &ts::Tensor<Element, 1>::template reshape<2>)
        .def("reshape1", &ts::Tensor<Element, 1>::template reshape<1>)

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
        .def_buffer([](ts::Tensor<Element, 1> &t) -> py::buffer_info {
            return py::buffer_info(t.data()->data(), sizeof(Element), py::format_descriptor<Element>::format(), 1,
                                   {t.shape(0)}, {sizeof(Element)});
        });
}

auto wrap_ops(pybind11::module &m)
{
    // ops_dot.hpp
    m.def("outer_product", &ts::outer_product);
    m.def("dot", py::overload_cast<ts::VectorF const &, ts::VectorF const &>(&ts::dot));
    m.def("dot", py::overload_cast<ts::MatrixF const &, ts::VectorF const &, bool>(&ts::dot));
    m.def("dot", py::overload_cast<ts::MatrixF const &, ts::MatrixF const &, bool, bool>(&ts::dot), py::arg("A"),
          py::arg("B"), py::arg("A_T") = false, py::arg("B_T") = false);
    m.def("dot", py::overload_cast<ts::Tensor<float, 3> const &, ts::MatrixF const &>(&ts::dot));

    // ops_common.hpp
    m.def("add_matrixf_matrixf", &ts::add<float, 2>);
    m.def("add_matrixi_matrixi", &ts::add<int, 2>);
    m.def("add_vectorf_vectorf", &ts::add<float, 1>);
    m.def("add_vectori_vectori", &ts::add<int, 1>);
    m.def("add_matrixf_vectorf", &ts::add<float>);
    m.def("add_matrixi_vectori", &ts::add<int>);

    m.def("multiply_vectorf_vectorf",
          py::overload_cast<ts::VectorF const &, ts::VectorF const &>(&ts::multiply<float, 1>));
    m.def("multiply_matrixf_matrixf",
          py::overload_cast<ts::MatrixF const &, ts::MatrixF const &>(&ts::multiply<float, 2>));
    m.def("multiply_vectorf_f", py::overload_cast<ts::VectorF const &, float>(&ts::multiply<float, 1>));
    m.def("multiply_matrixf_f", py::overload_cast<ts::MatrixF const &, float>(&ts::multiply<float, 2>));

    m.def("log", &ts::log<float, 2>);

    m.def("pow", &ts::pow<float, 2>);

    m.def("exp", &ts::exp<float, 2>);

    m.def("transpose", &ts::transpose);

    m.def("sum", py::overload_cast<ts::MatrixF const &, int>(&ts::sum_v2));

    m.def("get",
          [](ts::MatrixF const &m, ts::MatrixF const &i) { return ts::MatrixF({ts::get(m, i[0].cast<int>())}); });

    m.def("argmax_f", &ts::argmax<float>);
    m.def("argmax_i", &ts::argmax<int>);
}

template <typename Element, int Dim> auto wrap_nn_activations(pybind11::module &m, std::string postfix)
{
    py::class_<ts::ReLU<Element, Dim>>(m, ("ReLU" + postfix).c_str())
        .def(py::init<>())
        .def("__call__", &ts::ReLU<Element, Dim>::operator())
        .def("forward", &ts::ReLU<Element, Dim>::forward)
        .def("backward", &ts::ReLU<Element, Dim>::backward);
}

auto wrap_grad_holder(pybind11::module &m)
{
    py::class_<ts::GradHolder<float>, PyGradHolderFloat>(m, "GradHolderF")
        .def(py::init<>())
        .def("tensor", &ts::GradHolder<float>::tensor, py::return_value_policy::reference_internal)
        .def("grad", &ts::GradHolder<float>::grad, py::return_value_policy::reference_internal);

    py::class_<ts::GradHolder<int>, PyGradHolderInt>(m, "GradHolderI")
        .def(py::init<>())
        .def("tensor", &ts::GradHolder<int>::tensor, py::return_value_policy::reference_internal)
        .def("grad", &ts::GradHolder<int>::grad, py::return_value_policy::reference_internal);
}

template <typename Element, int Dim> auto wrap_variable(pybind11::module &m, char const *class_name)
{
    py::class_<ts::Variable<Element, Dim>, ts::GradHolder<Element>>(m, class_name)
        .def(py::init<std::array<size_type, Dim>>())
        .def("tensor", &ts::Variable<Element, Dim>::tensor, py::return_value_policy::reference_internal)
        .def("grad", &ts::Variable<Element, Dim>::grad, py::return_value_policy::reference_internal);
}

auto wrap_nn(pybind11::module &m)
{
    wrap_grad_holder(m);

    wrap_variable<float, 1>(m, "Variable1F");
    wrap_variable<float, 2>(m, "Variable2F");
    wrap_variable<float, 3>(m, "Variable3F");
    wrap_variable<float, 4>(m, "Variable4F");
    wrap_variable<int, 1>(m, "Variable1I");
    wrap_variable<int, 2>(m, "Variable2I");
    wrap_variable<int, 3>(m, "Variable3I");
    wrap_variable<int, 4>(m, "Variable4I");

    py::class_<ts::SGD<float>>(m, "SGD")
        .def(py::init<float, float>(), py::arg("lr"), py::arg("momentum") = 0.0)
        .def(py::init<std::vector<std::reference_wrapper<ts::GradHolder<float>>>, float, float>(), py::arg("params"), py::arg("lr"), py::arg("momentum") = 0.0)
        .def("step", &ts::SGD<float>::step)
        .def("register_params", py::overload_cast<ts::SGD<float>::VectorRef>(&ts::SGD<float>::register_params))
        .def("register_params", py::overload_cast<ts::SGD<float>::Ref>(&ts::SGD<float>::register_params));

    py::class_<ts::Adagrad<float>>(m, "Adagrad")
        .def(py::init<float>(), py::arg("lr"))
        .def(py::init<std::vector<std::reference_wrapper<ts::GradHolder<float>>>, float>(), py::arg("params"), py::arg("lr"))
        .def("step", &ts::Adagrad<float>::step)
        .def("register_params", py::overload_cast<ts::Adagrad<float>::VectorRef>(&ts::Adagrad<float>::register_params))
        .def("register_params", py::overload_cast<ts::Adagrad<float>::Ref>(&ts::Adagrad<float>::register_params));

    py::class_<ts::CrossEntropyLoss>(m, "CrossEntropyLoss")
        .def(py::init<>())
        .def("__call__", &ts::CrossEntropyLoss::operator())
        .def("forward", &ts::CrossEntropyLoss::forward)
        .def("backward", &ts::CrossEntropyLoss::backward);

    py::enum_<ts::Activation>(m, "Activation").value("RELU", ts::Activation::RELU).value("NONE", ts::Activation::NONE);

    wrap_nn_activations<float, 2>(m, "_f2");
    wrap_nn_activations<float, 3>(m, "_f3");

    py::class_<ts::LayerBase<float>>(m, "LayerBase")
        .def(py::init<>())
        .def("register_parameter", &ts::LayerBase<float>::register_parameter)
        .def("register_parameters", &ts::LayerBase<float>::register_parameters)
        .def("parameters", &ts::LayerBase<float>::parameters);

#ifdef TENSOR_USE_PROTOBUF
    py::class_<ts::Saver<float>>(m, "Saver")
        .def(py::init<ts::LayerBase<float> &>())
        .def("save", &ts::Saver<float>::save)
        .def("load", &ts::Saver<float>::load);
#endif

    py::class_<ts::FeedForward, ts::LayerBase<float>>(m, "FeedForward")
        .def(py::init(&ts::FeedForward::create))
        .def("__call__", &ts::FeedForward::operator())
        .def("forward", &ts::FeedForward::forward)
        .def("backward", &ts::FeedForward::backward)
        .def("bias", &ts::FeedForward::bias, py::return_value_policy::reference_internal)
        .def("weight", &ts::FeedForward::weight, py::return_value_policy::reference_internal)
        .def("weights", &ts::FeedForward::weights)
        .def("parameters", &ts::FeedForward::parameters);

    py::class_<ts::Conv2D, ts::LayerBase<float>>(m, "Conv2D")
        .def(py::init(&ts::Conv2D::create))
        .def("__call__", &ts::Conv2D::operator())
        .def("forward", &ts::Conv2D::forward)
        .def("backward", &ts::Conv2D::backward)
        .def("bias", &ts::Conv2D::bias, py::return_value_policy::reference_internal)
        .def("weight", &ts::Conv2D::weight, py::return_value_policy::reference_internal)
        .def("weights", &ts::Conv2D::weights)
        .def("parameters", &ts::Conv2D::parameters);

    py::class_<ts::MaxPool2D>(m, "MaxPool2D")
        .def(py::init(&ts::MaxPool2D::create))
        .def("__call__", &ts::MaxPool2D::operator())
        .def("forward", &ts::MaxPool2D::forward)
        .def("backward", &ts::MaxPool2D::backward);

    m.def("softmax", &ts::softmax);

    m.def("log_softmax", &ts::log_softmax);
}

PYBIND11_MODULE(libtensor, m)
{
    py::class_<ts::DataHolder<int>, PyDataHolderInt>(m, "DataHolderI").def(py::init<>());
    py::class_<ts::DataHolder<float>, PyDataHolderFloat>(m, "DataHolderF").def(py::init<>());

    wrap_tensor4D<int>(m, "Tensor4I");
    wrap_tensor4D<float>(m, "Tensor4F");
    wrap_tensor3D<int>(m, "Tensor3I");
    wrap_tensor3D<float>(m, "Tensor3F");
    wrap_tensor2D<int>(m, "MatrixI");
    wrap_tensor2D<float>(m, "MatrixF");
    wrap_tensor1D<int>(m, "VectorI");
    wrap_tensor1D<float>(m, "VectorF");
    wrap_ops(m);
    wrap_nn(m);
}
