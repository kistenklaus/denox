#pragma once

#include "RuntimeContext.hpp"
#include "denox/runtime.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h> // containers, string, etc.

namespace pydenox {

class RuntimeModel {
public:
  // Construct from an existing context and DNX payload (Program or bytes).
  explicit RuntimeModel(const RuntimeContext &context,
                        const py::object &program_or_bytes);
  ~RuntimeModel();

  // Introspection
  std::vector<std::string> input_names() const;
  std::vector<std::string> output_names() const;

  denox::DataType tensor_dtype(const std::string &name) const;
  denox::Layout tensor_layout(const std::string &name) const;

  // Specialize dynamic extents and create an instance.
  // Example: {"H":1080, "W":1920}
  RuntimeInstance
  create_instance(const py::dict &dynamic_extents = py::dict()) const;

  std::string repr() const;
};

}
