#include "denox/compiler.hpp"

int main() {
  denox::CompileOptions options;
  options.dnxVersion = 0;
  options.features.coopmat = denox::Require;
  options.features.fusion = denox::Require;
  options.features.memory_concat = denox::Enable;

  options.inputDescription.storage = denox::Storage::StorageBuffer;
  options.inputDescription.layout = denox::Layout::HWC;
  options.inputDescription.dtype = denox::DataType::Float16;
  options.outputDescription.storage = denox::Storage::StorageBuffer;
  options.outputDescription.layout = denox::Layout::HWC;
  options.outputDescription.dtype = denox::DataType::Float16;

  denox::compile("net.onnx", options);
}
