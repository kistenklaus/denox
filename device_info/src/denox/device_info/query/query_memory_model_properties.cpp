#include "denox/device_info/query/query_memory_model_properties.hpp"
#include "denox/device_info/ApiVersion.hpp"
#include "denox/device_info/query/has_device_extentsion.hpp"
#include <vulkan/vulkan.hpp>

namespace denox {

MemoryModelProperties
query_memory_model_properties(vk::Instance instance,
                              vk::PhysicalDevice physicalDevice,
                              ApiVersion apiVersion) {
  MemoryModelProperties out{};
  out.vmm = false;
  out.vmmDeviceScope = false;

  const bool apiAtLeast12 = apiVersion >= ApiVersion::VULKAN_1_2;

  const bool canQueryVmm =
      apiAtLeast12 ||
      has_device_extension(physicalDevice,
                           VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME);

  auto fpGetFeatures2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
      instance.getProcAddr("vkGetPhysicalDeviceFeatures2"));

  auto fpGetFeatures2KHR =
      reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2KHR>(
          instance.getProcAddr("vkGetPhysicalDeviceFeatures2KHR"));

  if (!fpGetFeatures2 && !fpGetFeatures2KHR) {
    return out;
  }

  vk::PhysicalDeviceFeatures2 features2{};

  vk::PhysicalDeviceVulkanMemoryModelFeatures vmm{};

  void **tail = &features2.pNext;

  if (canQueryVmm) {
    *tail = &vmm;
    tail = &vmm.pNext;
  }

  *tail = nullptr;

  if (fpGetFeatures2) {
    fpGetFeatures2(static_cast<VkPhysicalDevice>(physicalDevice),
                   reinterpret_cast<VkPhysicalDeviceFeatures2 *>(&features2));
  } else {
    fpGetFeatures2KHR(
        static_cast<VkPhysicalDevice>(physicalDevice),
        reinterpret_cast<VkPhysicalDeviceFeatures2 *>(&features2));
  }

  if (canQueryVmm) {
    out.vmm = static_cast<bool>(vmm.vulkanMemoryModel);
    out.vmmDeviceScope = static_cast<bool>(vmm.vulkanMemoryModelDeviceScope);
  }

  return out;
}

} // namespace denox
