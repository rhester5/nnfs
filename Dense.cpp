#include "Dense.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>


namespace py = pybind11;

PYBIND11_MODULE(dense, m) {
    py::class_<Dense<4, 3>>(m, "Dense")
        .def(py::init<const Eigen::Matrix<double, 4, 3>&, const Eigen::Vector<double, 3>&>())
        .def("set_weights", &Dense<4, 3>::setWeights)
        .def("set_biases", &Dense<4, 3>::setBiases)
        .def("forward", &Dense<4, 3>::forward<1>);
}
