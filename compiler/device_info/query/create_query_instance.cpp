#include "device_info/query/create_query_instance.hpp"
#include "device_info/ApiVersion.hpp"
#include "diag/unreachable.hpp"
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

namespace denox::compiler::device_info {

vk::Instance query::create_query_instance(ApiVersion apiVersion) {
  vk::ApplicationInfo appInfo;
  appInfo.pApplicationName = "denox-device-info-query";
  appInfo.applicationVersion = DENOX_VERSION;
  appInfo.engineVersion = DENOX_VERSION;
  appInfo.pEngineName = "denox";
  switch (apiVersion) {
  case ApiVersion::VULKAN_1_0:
    appInfo.apiVersion = VK_API_VERSION_1_0;
    break;
  case ApiVersion::VULKAN_1_1:
#ifdef VK_API_VERSION_1_1
    appInfo.apiVersion = VK_API_VERSION_1_1;
#else 
    appInfo.apiVersion = VK_API_VERSION_1_0;
#endif
    break;
  case ApiVersion::VULKAN_1_2:
#ifdef VK_API_VERSION_1_2
    appInfo.apiVersion = VK_API_VERSION_1_2;
#else
    appInfo.apiVersion = VK_API_VERSION_1_0;
#endif
    break;
  case ApiVersion::VULKAN_1_3:
#ifdef VK_API_VERSION_1_3
    appInfo.apiVersion = VK_API_VERSION_1_3;
#else
    appInfo.apiVersion = VK_API_VERSION_1_0;
#endif
    break;
  case ApiVersion::VULKAN_1_4:
#ifdef VK_API_VERSION_1_4
    appInfo.apiVersion = VK_API_VERSION_1_4;
#else 
    appInfo.apiVersion = VK_API_VERSION_1_0;
#endif
    break;
  default:
    compiler::diag::unreachable();
  }

  vk::InstanceCreateInfo createInfo;
  createInfo.pApplicationInfo = &appInfo;

  return vk::createInstance(createInfo);
}

} // namespace denox::compiler::device_info
