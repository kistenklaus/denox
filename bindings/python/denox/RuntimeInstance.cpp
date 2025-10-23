#include "RuntimeInstance.hpp"

namespace pydenox {

RuntimeInstance::RuntimeInstance(const RuntimeModel &model,
                                 const py::dict &dynamic_extents) {
  throw std::runtime_error("RuntimeInstance constructor is not implemented");
}

RuntimeInstance::~RuntimeInstance() { std::terminate(); }

py::tuple RuntimeInstance::tensor_shape(const std::string &name) const {
  throw std::runtime_error("RuntimeInstance tensor_shape is not implemented");
}

std::size_t RuntimeInstance::tensor_nbytes(const std::string &name) const {
  throw std::runtime_error("RuntimeInstance tensor_nbytes is not implemented");
}

py::dict RuntimeInstance::eval(const py::dict &inputs, py::object outputs) {
  throw std::runtime_error("RuntimeInstance eval is not implemented");
}

std::string RuntimeInstance::repr() const {
  throw std::runtime_error("RuntimeInstance repr is not implemented");
}

} // namespace pydenox
