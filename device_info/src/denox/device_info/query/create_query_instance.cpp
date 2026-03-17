#include "denox/device_info/query/create_query_instance.hpp"
#include "denox/device_info/ApiVersion.hpp"
#include "denox/diag/unreachable.hpp"
#include "denox/memory/container/vector.hpp"
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

namespace denox {

static uint32_t to_vk_version(ApiVersion v) {
  switch (v) {
  case ApiVersion::VULKAN_1_0: return VK_MAKE_API_VERSION(0, 1, 0, 0);
  case ApiVersion::VULKAN_1_1: return VK_MAKE_API_VERSION(0, 1, 1, 0);
  case ApiVersion::VULKAN_1_2: return VK_MAKE_API_VERSION(0, 1, 2, 0);
  case ApiVersion::VULKAN_1_3: return VK_MAKE_API_VERSION(0, 1, 3, 0);
  case ApiVersion::VULKAN_1_4: return VK_MAKE_API_VERSION(0, 1, 4, 0);
  default: diag::unreachable();
  }
}

static ApiVersion from_vk_version(uint32_t version) {
  const uint32_t major = VK_API_VERSION_MAJOR(version);
  const uint32_t minor = VK_API_VERSION_MINOR(version);

  if (major != 1) {
    diag::unreachable();
  }

  switch (minor) {
  case 0: return ApiVersion::VULKAN_1_0;
  case 1: return ApiVersion::VULKAN_1_1;
  case 2: return ApiVersion::VULKAN_1_2;
  case 3: return ApiVersion::VULKAN_1_3;
  case 4: return ApiVersion::VULKAN_1_4;
  default:
    diag::unreachable();
  }
}

vk::Instance create_query_instance(ApiVersion& apiVersion) {
  uint32_t loaderVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);

  auto pfnEnumerateInstanceVersion =
      reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
          vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion"));

  if (pfnEnumerateInstanceVersion) {
    pfnEnumerateInstanceVersion(&loaderVersion);
  }

  uint32_t requested = to_vk_version(apiVersion);
  uint32_t chosen = std::min(requested, loaderVersion);
  apiVersion = from_vk_version(chosen);

  vk::ApplicationInfo appInfo;
  appInfo.pApplicationName = "denox-device-info-query";
  appInfo.applicationVersion = DENOX_VERSION;
  appInfo.engineVersion = DENOX_VERSION;
  appInfo.pEngineName = "denox";
  appInfo.apiVersion = chosen;

  uint32_t count = 0;
  VkResult res = vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
  if (res != VK_SUCCESS) {
    throw std::runtime_error("vkEnumerateInstanceExtensionProperties failed");
  }

  memory::vector<VkExtensionProperties> available(count);
  res = vkEnumerateInstanceExtensionProperties(nullptr, &count, available.data());
  if (res != VK_SUCCESS && res != VK_INCOMPLETE) {
    throw std::runtime_error("vkEnumerateInstanceExtensionProperties failed");
  }

  const char* portabilityExt = nullptr;
  vk::InstanceCreateFlags flags{};

  for (const auto& ext : available) {
    if (std::strcmp(ext.extensionName,
                    VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0) {
      portabilityExt = VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
      flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
      break;
    }
  }

  vk::InstanceCreateInfo createInfo;
  createInfo.pApplicationInfo = &appInfo;
  createInfo.flags = flags;
  createInfo.enabledExtensionCount = portabilityExt ? 1u : 0u;
  createInfo.ppEnabledExtensionNames = portabilityExt ? &portabilityExt : nullptr;

  return vk::createInstance(createInfo);
}

} // namespace denox
