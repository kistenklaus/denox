#pragma once
#include "RuntimeContext.hpp"
#include "RuntimeModel.hpp"
#include "denox/runtime.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h> // containers, string, etc.
#include <pybind11/pybind11.h>

namespace pydenox {

class RuntimeInstance {
public:
  explicit RuntimeInstance(const RuntimeModel &model,
                           const py::dict &dynamic_extents = py::dict());
  ~RuntimeInstance();

  // Post-specialization shape and size queries
  // Returns (H, W, C) as Python tuple of ints.
  py::tuple tensor_shape(const std::string &name) const;

  // Total byte size for the named tensor in the current specialization.
  std::size_t tensor_nbytes(const std::string &name) const;

  // Run inference.
  //
  // inputs:  Mapping[str, tensor-like with __dlpack__] (typically CPU,
  // contiguous) outputs: Optional mapping[str, writable buffer/array] to write
  // into; if None, new arrays are created. return_format: "numpy" (default) or
  // "dlpack" (to return PyCapsules suitable for torch.from_dlpack).
  py::dict eval(const py::dict &inputs, py::object outputs = py::none());

  std::string repr() const;
};

} // namespace pydenox
