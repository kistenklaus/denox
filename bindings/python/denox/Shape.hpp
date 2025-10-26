#pragma once

#include <fmt/format.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace pydenox {

struct Shape {
  pybind11::object height;
  pybind11::object width;
  pybind11::object channels;

  Shape(pybind11::object h = pybind11::none(),
        pybind11::object w = pybind11::none(),
        pybind11::object c = pybind11::none())
      : height(std::move(h)), width(std::move(w)), channels(std::move(c)) {}

  std::string repr() const {
    auto to_str = [](const pybind11::object &o) {
      return pybind11::str(o).cast<std::string>();
    };
    return fmt::format("Shape(H={}, W={}, C={})", to_str(height), to_str(width),
                       to_str(channels));
  }

  static void define(pybind11::module_ &m) {
    pybind11::class_<Shape>(m, "Shape", "Represents a tensor shape (H, W, C).")
        .def(pybind11::init<pybind11::object, pybind11::object,
                            pybind11::object>(),
             pybind11::kw_only(), pybind11::arg("H") = pybind11::none(),
             pybind11::arg("W") = pybind11::none(),
             pybind11::arg("C") = pybind11::none(),
             "Create a new Shape(H, W, C). Each argument may be int, str, or "
             "None.")
        .def_readwrite("H", &Shape::height)
        .def_readwrite("W", &Shape::width)
        .def_readwrite("C", &Shape::channels)
        .def("__repr__", &Shape::repr);
  }
};

} // namespace pydenox
