#include "denox/compiler.hpp"
#include <CLI/App.hpp>

int main() {

  denox::CompileOptions options;
  options.dnxVersion = 0;
  options.features.coopmat = denox::Enable;
  options.features.fusion = denox::Enable;
  options.features.memory_concat = denox::Enable;

  options.inputDescription.storage = denox::Storage::StorageBuffer;
  options.inputDescription.layout = denox::Layout::HWC;
  options.inputDescription.dtype = denox::DataType::Float16;
  options.outputDescription.storage = denox::Storage::StorageBuffer;
  options.outputDescription.layout = denox::Layout::HWC;
  options.outputDescription.dtype = denox::DataType::Float16;

  options.device.deviceName = "*RTX*";
  options.device.apiVersion = denox::VulkanApiVersion::Vulkan_1_4;

  options.spirvOptions.debugInfo = false;
  options.spirvOptions.nonSemanticDebugInfo = false;
  options.spirvOptions.optimize = false;

  denox::compile("net.onnx", options);
}
