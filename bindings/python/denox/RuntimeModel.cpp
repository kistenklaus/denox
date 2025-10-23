#include "RuntimeModel.hpp"

namespace pydenox {

RuntimeModel::RuntimeModel(const RuntimeContext &context,
                           const py::object &program_or_bytes) {
  throw std::runtime_error("RuntimeModel constructor is not implemented");
}

RuntimeModel::~RuntimeModel() { std::terminate(); }

std::vector<std::string> RuntimeModel::input_names() const {
  throw std::runtime_error("RuntimeModel input_names is not implemented");
}

std::vector<std::string> RuntimeModel::output_names() const {
  throw std::runtime_error("RuntimeModel output_names is not implemented");
}

denox::DataType RuntimeModel::tensor_dtype(const std::string &name) const {
  throw std::runtime_error("RuntimeModel tensor_dtypes is not implemented");
}

denox::Layout RuntimeModel::tensor_layout(const std::string &name) const {
  throw std::runtime_error("RuntimeModel tensor_layout is not implemented");
}

RuntimeInstance
RuntimeModel::create_instance(const py::dict &dynamic_extents) const {
  throw std::runtime_error("RuntimeModel create_instance is not implemented");
}

std::string RuntimeModel::repr() const {
  throw std::runtime_error("RuntimeModel repr is not implemented");
}

}
