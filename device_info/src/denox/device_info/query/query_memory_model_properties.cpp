#include "denox/device_info/query/query_memory_model_properties.hpp"
#include "denox/device_info/ApiVersion.hpp"
#include "denox/device_info/query/has_device_extentsion.hpp"
#include <cstring>
#include <vulkan/vulkan.hpp>

namespace denox {

MemoryModelProperties
query_memory_model_properties(vk::Instance instance,
                              vk::PhysicalDevice physicalDevice,
                              ApiVersion apiVersion) {
  MemoryModelProperties out{};
  out.vmm = false;
  out.vmmDeviceScope = false;
  out.vmmAvailabilityVisibilityChains = false;
  out.bufferDeviceAddress = false;
  out.bufferDeviceAddressCaptureReplay = false;
  out.bufferDeviceAddressMultiDevice = false;

  const bool apiAtLeast12 = apiVersion >= ApiVersion::VULKAN_1_2;

  const bool canQueryVmm =
      apiAtLeast12 ||
      has_device_extension(physicalDevice,
                           VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME);

  const bool canQueryBda =
      apiAtLeast12 ||
      has_device_extension(physicalDevice,
                           VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);

  auto fpGetFeatures2 =
      reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
          instance.getProcAddr("vkGetPhysicalDeviceFeatures2"));

  auto fpGetFeatures2KHR =
      reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2KHR>(
          instance.getProcAddr("vkGetPhysicalDeviceFeatures2KHR"));

  if (!fpGetFeatures2 && !fpGetFeatures2KHR) {
    return out;
  }

  vk::PhysicalDeviceFeatures2 features2{};

  vk::PhysicalDeviceVulkanMemoryModelFeatures vmm{};
  vk::PhysicalDeviceBufferDeviceAddressFeatures bda{};

  void** tail = &features2.pNext;

  if (canQueryVmm) {
    *tail = &vmm;
    tail = &vmm.pNext;
  }

  if (canQueryBda) {
    *tail = &bda;
    tail = &bda.pNext;
  }

  *tail = nullptr;

  if (fpGetFeatures2) {
    fpGetFeatures2(static_cast<VkPhysicalDevice>(physicalDevice),
                   reinterpret_cast<VkPhysicalDeviceFeatures2*>(&features2));
  } else {
    fpGetFeatures2KHR(static_cast<VkPhysicalDevice>(physicalDevice),
                      reinterpret_cast<VkPhysicalDeviceFeatures2*>(&features2));
  }

  if (canQueryVmm) {
    out.vmm = static_cast<bool>(vmm.vulkanMemoryModel);
    out.vmmDeviceScope = static_cast<bool>(vmm.vulkanMemoryModelDeviceScope);
    out.vmmAvailabilityVisibilityChains =
        static_cast<bool>(vmm.vulkanMemoryModelAvailabilityVisibilityChains);
  }

  if (canQueryBda) {
    out.bufferDeviceAddress = static_cast<bool>(bda.bufferDeviceAddress);
    out.bufferDeviceAddressCaptureReplay =
        static_cast<bool>(bda.bufferDeviceAddressCaptureReplay);
    out.bufferDeviceAddressMultiDevice =
        static_cast<bool>(bda.bufferDeviceAddressMultiDevice);
  }

  return out;
}

} // namespace denox
