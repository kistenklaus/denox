#include "device_info/query/query_layout_rules.hpp"
#include <vulkan/vulkan.hpp>

namespace denox::compiler::device_info::query {

LayoutRules query_layout_rules([[maybe_unused]] vk::Instance instance,
                               vk::PhysicalDevice physicalDevice) {
  LayoutRules out;
  { // query scalar block layout
    vk::PhysicalDeviceScalarBlockLayoutFeatures scalarBlockFeatures{false};
    vk::PhysicalDeviceFeatures2 feature2;
    feature2.pNext = &scalarBlockFeatures;
    physicalDevice.getFeatures2(&feature2);
    out.scalarBlockLayout = scalarBlockFeatures.scalarBlockLayout;
  }
  {
    vk::PhysicalDeviceUniformBufferStandardLayoutFeatures uniformBlockLayout{
        false};
    vk::PhysicalDeviceFeatures2 feature2;
    feature2.pNext = &uniformBlockLayout;
    physicalDevice.getFeatures2(&feature2);
    out.uniformBufferStandardLayout =
        uniformBlockLayout.uniformBufferStandardLayout;
  }

  return out;
}

} // namespace denox::compiler::device_info::query
