#include "Module.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h> // containers, string, etc.

PYBIND11_MODULE(_denox, m) {
  m.doc() = "denox Python binding";
  pydenox::Module::define(m);
}
