#include "pybind11/pytypes.h"
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ranges>
#include <stdexcept>
#include <string_view>

namespace py = pybind11;

py::dict result_ok(py::object value) {
  py::dict result;
  result["value"] = value;
  result["error"] = py::none();
  return result;
}

py::dict result_err(const std::string_view msg) {
  py::dict result;
  result["value"] = py::none();
  result["error"] = py::str(msg);
  return result;
}

static py::dict generatePipeline(py::object obj) {
  return {};
}

PYBIND11_MODULE(_pyvk_cpp, m) {
  m.def("generatePipeline", &generatePipeline,
        "Build pipeline from reflected model");
}
