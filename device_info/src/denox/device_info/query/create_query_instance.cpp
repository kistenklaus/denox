#include "denox/device_info/query/create_query_instance.hpp"
#include "denox/device_info/ApiVersion.hpp"
#include "denox/diag/unreachable.hpp"
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

namespace denox {

vk::Instance create_query_instance(ApiVersion &apiVersion) {
  vk::ApplicationInfo appInfo;
  appInfo.pApplicationName = "denox-device-info-query";
  appInfo.applicationVersion = DENOX_VERSION;
  appInfo.engineVersion = DENOX_VERSION;
  appInfo.pEngineName = "denox";
  switch (apiVersion) {
  case ApiVersion::VULKAN_1_0:
    appInfo.apiVersion = VK_API_VERSION_1_0;
    apiVersion = ApiVersion::VULKAN_1_0;
    break;
  case ApiVersion::VULKAN_1_1:
#ifdef VK_API_VERSION_1_1
    appInfo.apiVersion = VK_API_VERSION_1_1;
    apiVersion = ApiVersion::VULKAN_1_1;
#else
    appInfo.apiVersion = VK_API_VERSION_1_0;
    apiVersion = ApiVersion::VULKAN_1_0;
#endif
    break;
  case ApiVersion::VULKAN_1_2:
#ifdef VK_API_VERSION_1_2
    appInfo.apiVersion = VK_API_VERSION_1_2;
    apiVersion = ApiVersion::VULKAN_1_2;
#else
    appInfo.apiVersion = VK_API_VERSION_1_0;
    apiVersion = ApiVersion::VULKAN_1_0;
#endif
    break;
  case ApiVersion::VULKAN_1_3:
#ifdef VK_API_VERSION_1_3
    appInfo.apiVersion = VK_API_VERSION_1_3;
    apiVersion = ApiVersion::VULKAN_1_3;
#else
    appInfo.apiVersion = VK_API_VERSION_1_0;
    apiVersion = ApiVersion::VULKAN_1_0;
#endif
    break;
  case ApiVersion::VULKAN_1_4:
#ifdef VK_API_VERSION_1_4
    appInfo.apiVersion = VK_API_VERSION_1_4;
    apiVersion = ApiVersion::VULKAN_1_4;
#else
    appInfo.apiVersion = VK_API_VERSION_1_0;
    apiVersion = ApiVersion::VULKAN_1_0;
#endif
    break;
  default:
    diag::unreachable();
  }

  vk::InstanceCreateInfo createInfo;
  createInfo.pApplicationInfo = &appInfo;
#ifdef __APPLE__
  const char *extensions[1]{VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME};
  createInfo.ppEnabledExtensionNames = extensions;
  createInfo.enabledExtensionCount = 1;
  createInfo.flags = vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
#else
#endif

  return vk::createInstance(createInfo);
}

} // namespace denox::compiler::device_info
