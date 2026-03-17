
#include "vulkan/vulkan.hpp"
#include <cstring>
namespace denox {

static inline bool has_device_extension(vk::PhysicalDevice physicalDevice, const char *name) {
  for (const auto &ext : physicalDevice.enumerateDeviceExtensionProperties()) {
    if (std::strcmp(ext.extensionName, name) == 0) {
      return true;
    }
  }
  return false;
}

}


