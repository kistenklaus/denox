#include "Module.hpp"
#include "denox/common/types.hpp"
#include "denox/compiler.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h> // containers, string, etc.

PYBIND11_MODULE(_denox, m) {
  m.doc() = "denox Python binding";

  pybind11::enum_<denox::Storage>(m, "Storage")
      .value("StorageBuffer", denox::Storage::StorageBuffer);

  pybind11::enum_<denox::Layout>(m, "Layout")
      .value("Undefined", denox::Layout::Undefined);

  pybind11::enum_<denox::DataType>(m, "DataType")
      .value("Auto", denox::DataType::Auto)
      .value("Float16", denox::DataType::Float16)
      .value("Float32", denox::DataType::Float32)
      .value("Uint8", denox::DataType::Uint8)
      .value("Int8", denox::DataType::Int8);

  pybind11::enum_<denox::Heuristic>(m, "Heuristic")
      .value("MemoryBandwidth", denox::Heuristic::MemoryBandwidth);

  pybind11::enum_<denox::VulkanApiVersion>(m, "TargetEnv")
      .value("Vulkan1_0", denox::VulkanApiVersion::Vulkan_1_0)
      .value("Vulkan1_1", denox::VulkanApiVersion::Vulkan_1_1)
      .value("Vulkan1_2", denox::VulkanApiVersion::Vulkan_1_2)
      .value("Vulkan1_3", denox::VulkanApiVersion::Vulkan_1_3)
      .value("Vulkan1_4", denox::VulkanApiVersion::Vulkan_1_4);

  pydenox::Tensor::define(m);
  pydenox::Shape::define(m);
  pydenox::Module::define(m);
}
