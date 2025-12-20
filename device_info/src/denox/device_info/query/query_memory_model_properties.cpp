#include "denox/device_info/query/query_memory_model_properties.hpp"

namespace denox {

MemoryModelProperties
query_model_model_properties([[maybe_unused]] vk::Instance instance,
                             vk::PhysicalDevice physicalDevice) {
  MemoryModelProperties out;
  { // query vmm support
    vk::PhysicalDeviceVulkanMemoryModelFeatures vmm;
    vk::PhysicalDeviceFeatures2 features2;
    features2.pNext = &vmm;
    physicalDevice.getFeatures2(&features2);
    out.vmm = vmm.vulkanMemoryModel;
    out.vmmDeviceScope = vmm.vulkanMemoryModelDeviceScope;
    out.vmmAvailabilityVisibilityChains =
        vmm.vulkanMemoryModelAvailabilityVisibilityChains;
  }
  { // query buffer address support
    vk::PhysicalDeviceBufferDeviceAddressFeatures bufferAddress;
    vk::PhysicalDeviceFeatures2 features2;
    features2.pNext = &bufferAddress;
    physicalDevice.getFeatures2(&features2);

    out.bufferDeviceAddress = bufferAddress.bufferDeviceAddress;
    out.bufferDeviceAddressCaptureReplay =
        bufferAddress.bufferDeviceAddressCaptureReplay;
    out.bufferDeviceAddressMultiDevice =
        bufferAddress.bufferDeviceAddressMultiDevice;
  }

  return out;
}

} // namespace denox::compiler::device_info::query
