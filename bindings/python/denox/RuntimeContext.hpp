#pragma once
#include "denox/runtime.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h> // containers, string, etc.

namespace pydenox {

class RuntimeContext {
public:
  // Create a runtime context on an optional device-name pattern (e.g.,
  // "*AMD*").
  explicit RuntimeContext(py::object device = py::none());
  ~RuntimeContext();

  // Context manager support (optional).
  RuntimeContext *enter();

  bool exit(py::object exc_type, py::object exc, py::object tb);

  // Create a RuntimeModel from either a Program or raw DNX bytes-like object.
  RuntimeModel load_model(const py::object &program_or_bytes) const;

  std::string repr() const;

private:
  void release();

  denox::RuntimeContext m_context;
};

}
