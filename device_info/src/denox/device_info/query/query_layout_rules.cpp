#include "denox/device_info/query/query_layout_rules.hpp"
#include "denox/device_info/ApiVersion.hpp"
#include "denox/device_info/query/has_device_extentsion.hpp"
#include <cstring>
#include <vulkan/vulkan.hpp>

namespace denox {

LayoutRules query_layout_rules(vk::Instance instance,
                               vk::PhysicalDevice physicalDevice,
                               ApiVersion apiVersion) {
  LayoutRules out{};
  out.scalarBlockLayout = false;
  out.uniformBufferStandardLayout = false;

  const bool apiAtLeast12 = apiVersion >= ApiVersion::VULKAN_1_2;

  const bool canQueryScalar =
      apiAtLeast12 ||
      has_device_extension(physicalDevice,
                           VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME);

  const bool canQueryUniformStd =
      apiAtLeast12 ||
      has_device_extension(
          physicalDevice, VK_KHR_UNIFORM_BUFFER_STANDARD_LAYOUT_EXTENSION_NAME);

  auto fpGetFeatures2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
      instance.getProcAddr("vkGetPhysicalDeviceFeatures2"));

  auto fpGetFeatures2KHR =
      reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2KHR>(
          instance.getProcAddr("vkGetPhysicalDeviceFeatures2KHR"));

  vk::PhysicalDeviceFeatures2 features2{};

  vk::PhysicalDeviceScalarBlockLayoutFeatures scalar{};
  vk::PhysicalDeviceUniformBufferStandardLayoutFeatures uniform{};

  void **tail = &features2.pNext;

  if (canQueryScalar) {
    *tail = &scalar;
    tail = &scalar.pNext;
  }

  if (canQueryUniformStd) {
    *tail = &uniform;
    tail = &uniform.pNext;
  }

  *tail = nullptr;

  if (fpGetFeatures2) {
    fpGetFeatures2(static_cast<VkPhysicalDevice>(physicalDevice),
                   reinterpret_cast<VkPhysicalDeviceFeatures2 *>(&features2));
  } else if (fpGetFeatures2KHR) {
    fpGetFeatures2KHR(
        static_cast<VkPhysicalDevice>(physicalDevice),
        reinterpret_cast<VkPhysicalDeviceFeatures2 *>(&features2));
  } else {
    // Vulkan 1.0 without VK_KHR_get_physical_device_properties2:
    // cannot query these extensible feature structs portably.
    return out;
  }

  if (canQueryScalar) {
    out.scalarBlockLayout = static_cast<bool>(scalar.scalarBlockLayout);
  }

  if (canQueryUniformStd) {
    out.uniformBufferStandardLayout =
        static_cast<bool>(uniform.uniformBufferStandardLayout);
  }

  return out;
}

} // namespace denox
