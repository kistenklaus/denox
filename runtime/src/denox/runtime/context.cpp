#include "context.hpp"
#include "denox/diag/unreachable.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <fmt/format.h>
#include <fmt/printf.h>
#include <stdexcept>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>

namespace denox::runtime {

// helpers
namespace {

static uint32_t to_vk_version(ApiVersion v) noexcept {
  switch (v) {
  case ApiVersion::VULKAN_1_0:
    return VK_MAKE_API_VERSION(0, 1, 0, 0);
  case ApiVersion::VULKAN_1_1:
    return VK_MAKE_API_VERSION(0, 1, 1, 0);
  case ApiVersion::VULKAN_1_2:
    return VK_MAKE_API_VERSION(0, 1, 2, 0);
  case ApiVersion::VULKAN_1_3:
    return VK_MAKE_API_VERSION(0, 1, 3, 0);
  case ApiVersion::VULKAN_1_4:
    return VK_MAKE_API_VERSION(0, 1, 4, 0);
  default:
    diag::unreachable();
  }
}

static ApiVersion from_vk_version(uint32_t v) {
  if (v >= VK_MAKE_API_VERSION(0, 1, 4, 0)) {
    return ApiVersion::VULKAN_1_4;
  }
  if (v >= VK_MAKE_API_VERSION(0, 1, 3, 0)) {
    return ApiVersion::VULKAN_1_3;
  }
  if (v >= VK_MAKE_API_VERSION(0, 1, 2, 0)) {
    return ApiVersion::VULKAN_1_2;
  }
  if (v >= VK_MAKE_API_VERSION(0, 1, 1, 0)) {
    return ApiVersion::VULKAN_1_1;
  }
  if (v >= VK_MAKE_API_VERSION(0, 1, 0, 0)) {
    return ApiVersion::VULKAN_1_0;
  }
  diag::unreachable();
}

static uint32_t query_loader_api_version() {
  uint32_t version = VK_MAKE_API_VERSION(0, 1, 0, 0);

  const auto pfnEnumerateInstanceVersion =
      reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
          vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion"));

  if (!pfnEnumerateInstanceVersion) {
    return version;
  }

  const VkResult result = pfnEnumerateInstanceVersion(&version);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("vkEnumerateInstanceVersion failed");
  }

  return version;
}

static uint32_t clamp_instance_api_version_raw(uint32_t requested) {
  return std::min(requested, query_loader_api_version());
}

static ApiVersion clamp_instance_api_version(ApiVersion requested) {
  return from_vk_version(
      clamp_instance_api_version_raw(to_vk_version(requested)));
}

} // namespace

// #define DENOX_QUITE

static VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              [[maybe_unused]] VkDebugUtilsMessageTypeFlagsEXT messageType,
              const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
              [[maybe_unused]] void *pUserData) {
  enum class Severity {
    None,
    Verbose,
    Info,
    Warning,
    Error,
  };
  Severity severity = Severity::None;
  if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
    severity = Severity::Verbose;
  }
  if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
    severity = Severity::Info;
  }
  if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    severity = Severity::Warning;
  }
  if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
    severity = Severity::Error;
  }
  // return VK_FALSE;
  switch (severity) {
  case Severity::None:
    fmt::println("[Validation-Layer]: {}", pCallbackData->pMessage);
    break;
  case Severity::Verbose:
    fmt::println("\x1B[37m[Validation-Layer]:\x1B[0m {}",
                 pCallbackData->pMessage);
    break;
  case Severity::Info:
    fmt::println("\x1B[34m[Validation-Layer]:\x1B[0m {}",
                 pCallbackData->pMessage);
    break;
  case Severity::Warning:
    fmt::println("\x1B[33m[Validation-Layer]:\x1B[0m\n{}",
                 pCallbackData->pMessage);
    break;
  case Severity::Error:
    fmt::println("\x1B[31m[Validation-Layer]:\x1B[0m\n{}",
                 pCallbackData->pMessage);
    break;
  }

  return VK_FALSE;
}

VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDebugUtilsMessengerEXT *pDebugMessenger) {
  auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
      vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
  if (func != nullptr) {
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks *pAllocator) {
  auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
      vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
  if (func != nullptr) {
    func(instance, debugMessenger, pAllocator);
  }
}

static bool checkLayerSupport(const char *layerName) {
  uint32_t layerCount;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
  memory::vector<VkLayerProperties> layers(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, layers.data());
  return std::ranges::find_if(
             layers, [layerName](const VkLayerProperties &layer) {
               return std::strcmp(layer.layerName, layerName) == 0;
             }) != layers.end();
}

// ASCII lowercase
static std::string to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

// Simple case-insensitive glob: '*' and '?' only.
static bool glob_match_ci(std::string pat, std::string text) {
  pat = to_lower(std::move(pat));
  text = to_lower(std::move(text));

  size_t p = 0, t = 0, star = std::string::npos, match = 0;
  while (t < text.size()) {
    if (p < pat.size() && (pat[p] == '?' || pat[p] == text[t])) {
      ++p;
      ++t;
    } else if (p < pat.size() && pat[p] == '*') {
      star = p++;
      match = t; // remember where '*' is and the match start
    } else if (star != std::string::npos) {
      p = star + 1;
      t = ++match; // backtrack: let '*' eat one more char
    } else {
      return false;
    }
  }
  while (p < pat.size() && pat[p] == '*')
    ++p;
  return p == pat.size();
}

struct ComputeQueueSelection {
  uint32_t family = VK_QUEUE_FAMILY_IGNORED;
};

static ComputeQueueSelection
pick_best_compute_queue_family(VkPhysicalDevice phys) {
  uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, nullptr);
  memory::vector<VkQueueFamilyProperties> props(count);
  vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, props.data());

  int bestScore = std::numeric_limits<int>::min();
  uint32_t bestIdx = VK_QUEUE_FAMILY_IGNORED;

  for (uint32_t i = 0; i < count; ++i) {
    const auto &p = props[i];
    if (!(p.queueFlags & VK_QUEUE_COMPUTE_BIT))
      continue; // must be compute-capable
    if (p.queueCount == 0)
      continue;

    // Heuristic: prefer compute-only, then anything compute-capable.
    int score = 0;
    // Strong base if it’s compute-capable
    score += 1000;
    // Prefer families without graphics (often map to async compute on some
    // vendors)
    if (!(p.queueFlags & VK_QUEUE_GRAPHICS_BIT))
      score += 100;
    // Timestamp support is nice for profiling (not performance), tiny
    // tiebreaker
    if (p.timestampValidBits > 0)
      score += 1;

    // (Optional) prefer larger queue counts to avoid contention if you ever
    // grow score += static_cast<int>(p.queueCount);

    if (score > bestScore) {
      bestScore = score;
      bestIdx = i;
    }
  }

  return {bestIdx};
}

Context::Context(const char *deviceName, ApiVersion target_env,
                 bool enableValidationLayer)
    : m_instance(VK_NULL_HANDLE), m_device(VK_NULL_HANDLE),
      m_physicalDevice(VK_NULL_HANDLE), m_queue(VK_NULL_HANDLE) {

  m_debugMessenger = VK_NULL_HANDLE;
  m_vma = VK_NULL_HANDLE;
  m_support = {};

  // Assumes DeviceSupport has these two raw version fields.
  // If your struct only has one field, keep instanceApiVersion local and store
  // only the effective device version in m_support.apiVersion.
  m_support.instanceApiVersion =
      clamp_instance_api_version_raw(to_vk_version(target_env));

  { // Create instance.
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pNext = nullptr;
    appInfo.apiVersion = m_support.instanceApiVersion;
    appInfo.applicationVersion = DENOX_VERSION;
    appInfo.pApplicationName = "denox";
    appInfo.engineVersion = DENOX_VERSION;
    appInfo.pEngineName = "denox";

    uint32_t instanceExtensionCount = 0;
    VkResult result = vkEnumerateInstanceExtensionProperties(
        nullptr, &instanceExtensionCount, nullptr);
    if (result != VK_SUCCESS) {
      throw std::runtime_error(
          "Failed to enumerate Vulkan instance extensions.");
    }

    memory::vector<VkExtensionProperties> instanceExtensions(
        instanceExtensionCount);
    result = vkEnumerateInstanceExtensionProperties(
        nullptr, &instanceExtensionCount, instanceExtensions.data());
    if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
      throw std::runtime_error(
          "Failed to enumerate Vulkan instance extensions.");
    }

    bool hasPortabilityEnumeration = false;
    bool hasDebugUtils = false;
    for (uint32_t i = 0; i < instanceExtensionCount; ++i) {
      if (std::strcmp(instanceExtensions[i].extensionName,
                      VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0) {
        hasPortabilityEnumeration = true;
      }
      if (std::strcmp(instanceExtensions[i].extensionName,
                      VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0) {
        hasDebugUtils = true;
      }
    }

    memory::vector<const char *> enabledInstanceExtensions;
    if (hasPortabilityEnumeration) {
      enabledInstanceExtensions.push_back(
          VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    }

    memory::vector<const char *> enabledLayers;

    VkDebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfo{};
    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pNext = nullptr;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.flags = 0;

    if (hasPortabilityEnumeration) {
      createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }

    if (enableValidationLayer) {
      uint32_t layerCount = 0;
      result = vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
      if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to enumerate Vulkan instance layers.");
      }

      memory::vector<VkLayerProperties> layers(layerCount);
      result = vkEnumerateInstanceLayerProperties(&layerCount, layers.data());
      if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
        throw std::runtime_error("Failed to enumerate Vulkan instance layers.");
      }

      bool validationLayerSupported = false;
      for (uint32_t i = 0; i < layerCount; ++i) {
        if (std::strcmp(layers[i].layerName, "VK_LAYER_KHRONOS_validation") ==
            0) {
          validationLayerSupported = true;
          break;
        }
      }

      if (validationLayerSupported && hasDebugUtils) {
        enabledLayers.push_back("VK_LAYER_KHRONOS_validation");
        enabledInstanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

        debugUtilsMessengerCreateInfo.sType =
            VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugUtilsMessengerCreateInfo.pNext = nullptr;
        debugUtilsMessengerCreateInfo.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugUtilsMessengerCreateInfo.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugUtilsMessengerCreateInfo.pfnUserCallback = debugCallback;
        debugUtilsMessengerCreateInfo.pUserData = nullptr;

        createInfo.pNext = &debugUtilsMessengerCreateInfo;
      } else {
        enableValidationLayer = false;
      }
    }

    createInfo.ppEnabledExtensionNames = enabledInstanceExtensions.data();
    createInfo.enabledExtensionCount =
        static_cast<uint32_t>(enabledInstanceExtensions.size());
    createInfo.ppEnabledLayerNames = enabledLayers.data();
    createInfo.enabledLayerCount = static_cast<uint32_t>(enabledLayers.size());

    result = vkCreateInstance(&createInfo, nullptr, &m_instance);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to create Vulkan instance.");
    }

    if (enableValidationLayer) {
      result = CreateDebugUtilsMessengerEXT(m_instance,
                                            &debugUtilsMessengerCreateInfo,
                                            nullptr, &m_debugMessenger);
      if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create debug utils messenger.");
      }
    }
  }

  { // Select physical device.
    uint32_t deviceCount = 0;
    VkResult result =
        vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to enumerate Vulkan physical devices.");
    }

    memory::vector<VkPhysicalDevice> physicalDevices(deviceCount);
    result = vkEnumeratePhysicalDevices(m_instance, &deviceCount,
                                        physicalDevices.data());
    if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
      throw std::runtime_error("Failed to enumerate Vulkan physical devices.");
    }

    if (physicalDevices.empty()) {
      throw std::runtime_error(
          "Failed to select physical device: No Vulkan device found.");
    }

    if (deviceName == nullptr) {
      m_physicalDevice = physicalDevices[0];
      for (VkPhysicalDevice d : physicalDevices) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(d, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
          m_physicalDevice = d;
          break;
        }
      }
    } else {
      std::string devicePattern(deviceName);

      memory::vector<VkPhysicalDevice> matches;
      for (VkPhysicalDevice d : physicalDevices) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(d, &props);
        if (glob_match_ci(devicePattern, props.deviceName)) {
          matches.push_back(d);
        }
      }

      if (matches.empty()) {
        throw std::runtime_error(fmt::format(
            "Failed to select physical device: pattern \"{}\" did not "
            "match any device.",
            devicePattern));
      }

      if (matches.size() > 1) {
        std::string list;
        for (VkPhysicalDevice d : matches) {
          VkPhysicalDeviceProperties props{};
          vkGetPhysicalDeviceProperties(d, &props);
          list += std::string(props.deviceName) + "; ";
        }
        throw std::runtime_error(
            fmt::format("Failed to select physical device: pattern \"{}\" is "
                        "ambiguous, matches multiple devices: {}",
                        devicePattern, list));
      }

      m_physicalDevice = matches.front();
    }
  }

  VkPhysicalDeviceProperties deviceProperties{};
  vkGetPhysicalDeviceProperties(m_physicalDevice, &deviceProperties);
  m_timestampPeriod = deviceProperties.limits.timestampPeriod;
  m_support.deviceApiVersion =
      std::min(m_support.instanceApiVersion, deviceProperties.apiVersion);

  uint32_t deviceExtensionCount = 0;
  {
    VkResult result = vkEnumerateDeviceExtensionProperties(
        m_physicalDevice, nullptr, &deviceExtensionCount, nullptr);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to enumerate Vulkan device extensions.");
    }
  }

  memory::vector<VkExtensionProperties> deviceExtensions(deviceExtensionCount);
  {
    VkResult result = vkEnumerateDeviceExtensionProperties(
        m_physicalDevice, nullptr, &deviceExtensionCount,
        deviceExtensions.data());
    if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
      throw std::runtime_error("Failed to enumerate Vulkan device extensions.");
    }
  }

  bool has16BitStorageExt = false;
  bool hasShaderFloat16Int8Ext = false;
  bool hasVulkanMemoryModelExt = false;
  bool hasBufferDeviceAddressExt = false;
  bool hasSubgroupSizeControlExt = false;
  bool hasCooperativeMatrixExt = false;
  bool hasPortabilitySubsetExt = false;
  bool hasPciBusInfoExt = false;
  bool hasGetMemoryRequirements2Ext = false;
  bool hasDedicatedAllocationExt = false;
  bool hasBindMemory2Ext = false;
  bool hasMaintenance4Ext = false;
  bool hasMaintenance5Ext = false;
  bool hasMemoryBudgetExt = false;
  bool hasMemoryPriorityExt = false;
  bool hasAmdDeviceCoherentMemoryExt = false;
  bool hasExternalMemoryWin32Ext = false;
  uint32_t subgroupSizeControlExtVersion = 0;

  for (uint32_t i = 0; i < deviceExtensionCount; ++i) {
    const VkExtensionProperties &ext = deviceExtensions[i];

    if (std::strcmp(ext.extensionName, VK_KHR_16BIT_STORAGE_EXTENSION_NAME) ==
        0) {
      has16BitStorageExt = true;
    } else if (std::strcmp(ext.extensionName,
                           VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME) == 0) {
      hasShaderFloat16Int8Ext = true;
    } else if (std::strcmp(ext.extensionName,
                           VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME) == 0) {
      hasVulkanMemoryModelExt = true;
    } else if (std::strcmp(ext.extensionName,
                           VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) == 0) {
      hasBufferDeviceAddressExt = true;
    } else if (std::strcmp(ext.extensionName,
                           VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME) == 0) {
      hasSubgroupSizeControlExt = true;
      subgroupSizeControlExtVersion = ext.specVersion;
    } else if (std::strcmp(ext.extensionName,
                           VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0) {
      hasCooperativeMatrixExt = true;
    } else if (std::strcmp(ext.extensionName, "VK_KHR_portability_subset") ==
               0) {
      hasPortabilitySubsetExt = true;
    } else if (std::strcmp(ext.extensionName,
                           VK_EXT_PCI_BUS_INFO_EXTENSION_NAME) == 0) {
      hasPciBusInfoExt = true;
    } else if (std::strcmp(ext.extensionName,
                           VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME) ==
               0) {
      hasGetMemoryRequirements2Ext = true;
    } else if (std::strcmp(ext.extensionName,
                           VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME) == 0) {
      hasDedicatedAllocationExt = true;
    } else if (std::strcmp(ext.extensionName,
                           VK_KHR_BIND_MEMORY_2_EXTENSION_NAME) == 0) {
      hasBindMemory2Ext = true;
    } else if (std::strcmp(ext.extensionName,
                           VK_KHR_MAINTENANCE_4_EXTENSION_NAME) == 0) {
      hasMaintenance4Ext = true;
    } else if (std::strcmp(ext.extensionName,
                           VK_KHR_MAINTENANCE_5_EXTENSION_NAME) == 0) {
      hasMaintenance5Ext = true;
    } else if (std::strcmp(ext.extensionName,
                           VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0) {
      hasMemoryBudgetExt = true;
    } else if (std::strcmp(ext.extensionName,
                           VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME) == 0) {
      hasMemoryPriorityExt = true;
    } else if (std::strcmp(ext.extensionName,
                           VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME) == 0) {
      hasAmdDeviceCoherentMemoryExt = true;
    }
#if defined(_WIN32)
    else if (std::strcmp(ext.extensionName,
                         VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME) == 0) {
      hasExternalMemoryWin32Ext = true;
    }
#endif
  }

  m_support.portabilitySubset = hasPortabilitySubsetExt;
  m_support.cooperativeMatrixExt = hasCooperativeMatrixExt;
  m_support.subgroupSizeControlExt = hasSubgroupSizeControlExt;
  m_support.subgroupSizeControlExtVersion = subgroupSizeControlExtVersion;
  m_support.pciBusInfo = hasPciBusInfoExt;
  m_support.memoryBudget = hasMemoryBudgetExt;
  m_support.memoryPriority = hasMemoryPriorityExt;
  m_support.bufferDeviceAddressExt = hasBufferDeviceAddressExt;

  m_support.dedicatedAllocation =
      (m_support.deviceApiVersion >= VK_MAKE_API_VERSION(0, 1, 1, 0)) ||
      (hasGetMemoryRequirements2Ext && hasDedicatedAllocationExt);
  m_support.bindMemory2 =
      (m_support.deviceApiVersion >= VK_MAKE_API_VERSION(0, 1, 1, 0)) ||
      hasBindMemory2Ext;
  m_support.maintenance4 =
      (m_support.deviceApiVersion >= VK_MAKE_API_VERSION(0, 1, 3, 0)) ||
      hasMaintenance4Ext;
  m_support.maintenance5 =
      (m_support.deviceApiVersion >= VK_MAKE_API_VERSION(0, 1, 4, 0)) ||
      hasMaintenance5Ext;

  VkPhysicalDeviceFeatures availableBaseFeatures{};
  vkGetPhysicalDeviceFeatures(m_physicalDevice, &availableBaseFeatures);
  m_support.robustBufferAccess =
      (availableBaseFeatures.robustBufferAccess == VK_TRUE);
  m_support.shaderInt16 = (availableBaseFeatures.shaderInt16 == VK_TRUE);

  PFN_vkGetPhysicalDeviceFeatures2 fpGetPhysicalDeviceFeatures2 =
      reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
          vkGetInstanceProcAddr(m_instance, "vkGetPhysicalDeviceFeatures2"));
  PFN_vkGetPhysicalDeviceFeatures2KHR fpGetPhysicalDeviceFeatures2KHR =
      reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2KHR>(
          vkGetInstanceProcAddr(m_instance, "vkGetPhysicalDeviceFeatures2KHR"));

  if (fpGetPhysicalDeviceFeatures2 != nullptr ||
      fpGetPhysicalDeviceFeatures2KHR != nullptr) {
    VkPhysicalDeviceFeatures2 queryFeatures2{};
    queryFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    queryFeatures2.pNext = nullptr;

    VkPhysicalDevice16BitStorageFeatures query16BitStorage{};
    query16BitStorage.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    query16BitStorage.pNext = nullptr;

    VkPhysicalDeviceShaderFloat16Int8Features queryFloat16Int8{};
    queryFloat16Int8.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    queryFloat16Int8.pNext = nullptr;

    VkPhysicalDeviceVulkanMemoryModelFeatures queryMemoryModel{};
    queryMemoryModel.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES;
    queryMemoryModel.pNext = nullptr;

    VkPhysicalDeviceBufferDeviceAddressFeatures queryBufferDeviceAddress{};
    queryBufferDeviceAddress.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    queryBufferDeviceAddress.pNext = nullptr;

    VkPhysicalDeviceSubgroupSizeControlFeatures querySubgroupSizeControl{};
    querySubgroupSizeControl.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
    querySubgroupSizeControl.pNext = nullptr;

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR queryCooperativeMatrix{};
    queryCooperativeMatrix.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    queryCooperativeMatrix.pNext = nullptr;

    bool query16BitStorageSupported =
        (m_support.deviceApiVersion >= VK_MAKE_API_VERSION(0, 1, 1, 0)) ||
        has16BitStorageExt;
    bool queryFloat16Int8Supported =
        (m_support.deviceApiVersion >= VK_MAKE_API_VERSION(0, 1, 2, 0)) ||
        hasShaderFloat16Int8Ext;
    bool queryMemoryModelSupported =
        (m_support.deviceApiVersion >= VK_MAKE_API_VERSION(0, 1, 2, 0)) ||
        hasVulkanMemoryModelExt;
    bool queryBufferDeviceAddressSupported =
        (m_support.deviceApiVersion >= VK_MAKE_API_VERSION(0, 1, 2, 0)) ||
        hasBufferDeviceAddressExt;
    bool querySubgroupSizeControlSupported =
        (m_support.deviceApiVersion >= VK_MAKE_API_VERSION(0, 1, 3, 0)) ||
        hasSubgroupSizeControlExt;
    bool queryCooperativeMatrixSupported = hasCooperativeMatrixExt;

    void **queryTail = &queryFeatures2.pNext;

    if (query16BitStorageSupported) {
      *queryTail = &query16BitStorage;
      queryTail = &query16BitStorage.pNext;
    }
    if (queryFloat16Int8Supported) {
      *queryTail = &queryFloat16Int8;
      queryTail = &queryFloat16Int8.pNext;
    }
    if (queryMemoryModelSupported) {
      *queryTail = &queryMemoryModel;
      queryTail = &queryMemoryModel.pNext;
    }
    if (queryBufferDeviceAddressSupported) {
      *queryTail = &queryBufferDeviceAddress;
      queryTail = &queryBufferDeviceAddress.pNext;
    }
    if (querySubgroupSizeControlSupported) {
      *queryTail = &querySubgroupSizeControl;
      queryTail = &querySubgroupSizeControl.pNext;
    }
    if (queryCooperativeMatrixSupported) {
      *queryTail = &queryCooperativeMatrix;
      queryTail = &queryCooperativeMatrix.pNext;
    }
    *queryTail = nullptr;

    if (fpGetPhysicalDeviceFeatures2 != nullptr) {
      fpGetPhysicalDeviceFeatures2(m_physicalDevice, &queryFeatures2);
    } else {
      fpGetPhysicalDeviceFeatures2KHR(m_physicalDevice, &queryFeatures2);
    }

    if (query16BitStorageSupported) {
      m_support.storageBuffer16BitAccess =
          (query16BitStorage.storageBuffer16BitAccess == VK_TRUE);
    }
    if (queryFloat16Int8Supported) {
      m_support.shaderFloat16 = (queryFloat16Int8.shaderFloat16 == VK_TRUE);
    }
    if (queryMemoryModelSupported) {
      m_support.vulkanMemoryModel =
          (queryMemoryModel.vulkanMemoryModel == VK_TRUE);
      m_support.vulkanMemoryModelDeviceScope =
          (queryMemoryModel.vulkanMemoryModelDeviceScope == VK_TRUE);
    }
    if (queryBufferDeviceAddressSupported) {
      m_support.bufferDeviceAddress =
          (queryBufferDeviceAddress.bufferDeviceAddress == VK_TRUE);
      m_support.bufferDeviceAddressCaptureReplay =
          (queryBufferDeviceAddress.bufferDeviceAddressCaptureReplay ==
           VK_TRUE);
      m_support.bufferDeviceAddressMultiDevice =
          (queryBufferDeviceAddress.bufferDeviceAddressMultiDevice == VK_TRUE);
    }
    if (querySubgroupSizeControlSupported) {
      m_support.subgroupSizeControl =
          (querySubgroupSizeControl.subgroupSizeControl == VK_TRUE);
      m_support.computeFullSubgroups =
          (querySubgroupSizeControl.computeFullSubgroups == VK_TRUE);
    }
    if (queryCooperativeMatrixSupported) {
      m_support.cooperativeMatrix =
          (queryCooperativeMatrix.cooperativeMatrix == VK_TRUE);
    }

    if (hasSubgroupSizeControlExt && subgroupSizeControlExtVersion < 2 &&
        m_support.deviceApiVersion < VK_MAKE_API_VERSION(0, 1, 3, 0)) {
      m_support.subgroupSizeControl = true;
      m_support.computeFullSubgroups = true;
    }
  }

  { // Create logical device and compute queue.
    ComputeQueueSelection sel =
        pick_best_compute_queue_family(m_physicalDevice);
    m_queueFamily = sel.family;

    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.pNext = nullptr;
    queueCreateInfo.flags = 0;
    float queuePriority = 1.0f;
    queueCreateInfo.pQueuePriorities = &queuePriority;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.queueFamilyIndex = sel.family;

    memory::vector<const char *> enabledDeviceExtensions;

    if (hasPortabilitySubsetExt) {
      enabledDeviceExtensions.push_back("VK_KHR_portability_subset");
    }

    if (hasPciBusInfoExt) {
      enabledDeviceExtensions.push_back(VK_EXT_PCI_BUS_INFO_EXTENSION_NAME);
    }

    if (m_support.deviceApiVersion < VK_MAKE_API_VERSION(0, 1, 1, 0)) {
      if (m_support.storageBuffer16BitAccess && has16BitStorageExt) {
        enabledDeviceExtensions.push_back(VK_KHR_16BIT_STORAGE_EXTENSION_NAME);
      }
      if (m_support.dedicatedAllocation && hasGetMemoryRequirements2Ext &&
          hasDedicatedAllocationExt) {
        enabledDeviceExtensions.push_back(
            VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
        enabledDeviceExtensions.push_back(
            VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
      }
      if (m_support.bindMemory2 && hasBindMemory2Ext) {
        enabledDeviceExtensions.push_back(VK_KHR_BIND_MEMORY_2_EXTENSION_NAME);
      }
    }

    if (m_support.deviceApiVersion < VK_MAKE_API_VERSION(0, 1, 2, 0)) {
      if (m_support.shaderFloat16 && hasShaderFloat16Int8Ext) {
        enabledDeviceExtensions.push_back(
            VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
      }
      if (m_support.vulkanMemoryModel && hasVulkanMemoryModelExt) {
        enabledDeviceExtensions.push_back(
            VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME);
      }
      if (m_support.bufferDeviceAddress && hasBufferDeviceAddressExt) {
        enabledDeviceExtensions.push_back(
            VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
      }
    }

    if (m_support.deviceApiVersion < VK_MAKE_API_VERSION(0, 1, 3, 0)) {
      if (m_support.subgroupSizeControl && hasSubgroupSizeControlExt) {
        enabledDeviceExtensions.push_back(
            VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
      }
      if (m_support.maintenance4 && hasMaintenance4Ext) {
        enabledDeviceExtensions.push_back(VK_KHR_MAINTENANCE_4_EXTENSION_NAME);
      }
    }

    if (m_support.deviceApiVersion < VK_MAKE_API_VERSION(0, 1, 4, 0)) {
      if (m_support.maintenance5 && hasMaintenance5Ext) {
        enabledDeviceExtensions.push_back(VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
      }
    }

    if (hasMemoryBudgetExt) {
      enabledDeviceExtensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    }
    if (hasMemoryPriorityExt) {
      enabledDeviceExtensions.push_back(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME);
    }
    if (hasAmdDeviceCoherentMemoryExt) {
      enabledDeviceExtensions.push_back(
          VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME);
    }
#if defined(_WIN32)
    if (hasExternalMemoryWin32Ext) {
      enabledDeviceExtensions.push_back(
          VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    }
#endif
    if (m_support.cooperativeMatrix && hasCooperativeMatrixExt) {
      enabledDeviceExtensions.push_back(
          VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    }

    VkPhysicalDeviceFeatures enabledBaseFeatures{};
    enabledBaseFeatures.robustBufferAccess =
        m_support.robustBufferAccess ? VK_TRUE : VK_FALSE;
    enabledBaseFeatures.shaderInt16 =
        m_support.shaderInt16 ? VK_TRUE : VK_FALSE;

    VkPhysicalDevice16BitStorageFeatures enable16BitStorage{};
    enable16BitStorage.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    enable16BitStorage.pNext = nullptr;

    VkPhysicalDeviceShaderFloat16Int8Features enableFloat16Int8{};
    enableFloat16Int8.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    enableFloat16Int8.pNext = nullptr;

    VkPhysicalDeviceVulkanMemoryModelFeatures enableMemoryModel{};
    enableMemoryModel.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES;
    enableMemoryModel.pNext = nullptr;

    VkPhysicalDeviceBufferDeviceAddressFeatures enableBufferDeviceAddress{};
    enableBufferDeviceAddress.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    enableBufferDeviceAddress.pNext = nullptr;

    VkPhysicalDeviceSubgroupSizeControlFeatures enableSubgroupSizeControl{};
    enableSubgroupSizeControl.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
    enableSubgroupSizeControl.pNext = nullptr;

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR enableCooperativeMatrix{};
    enableCooperativeMatrix.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    enableCooperativeMatrix.pNext = nullptr;

    void *pNextDevice = nullptr;

    if (m_support.storageBuffer16BitAccess) {
      enable16BitStorage.storageBuffer16BitAccess = VK_TRUE;
      enable16BitStorage.pNext = pNextDevice;
      pNextDevice = &enable16BitStorage;
    }

    if (m_support.shaderFloat16) {
      enableFloat16Int8.shaderFloat16 = VK_TRUE;
      enableFloat16Int8.pNext = pNextDevice;
      pNextDevice = &enableFloat16Int8;
    }

    if (m_support.vulkanMemoryModel || m_support.vulkanMemoryModelDeviceScope) {
      enableMemoryModel.vulkanMemoryModel =
          m_support.vulkanMemoryModel ? VK_TRUE : VK_FALSE;
      enableMemoryModel.vulkanMemoryModelDeviceScope =
          m_support.vulkanMemoryModelDeviceScope ? VK_TRUE : VK_FALSE;
      enableMemoryModel.pNext = pNextDevice;
      pNextDevice = &enableMemoryModel;
    }

    if (m_support.bufferDeviceAddress ||
        m_support.bufferDeviceAddressCaptureReplay ||
        m_support.bufferDeviceAddressMultiDevice) {
      enableBufferDeviceAddress.bufferDeviceAddress =
          m_support.bufferDeviceAddress ? VK_TRUE : VK_FALSE;
      enableBufferDeviceAddress.bufferDeviceAddressCaptureReplay =
          m_support.bufferDeviceAddressCaptureReplay ? VK_TRUE : VK_FALSE;
      enableBufferDeviceAddress.bufferDeviceAddressMultiDevice =
          m_support.bufferDeviceAddressMultiDevice ? VK_TRUE : VK_FALSE;
      enableBufferDeviceAddress.pNext = pNextDevice;
      pNextDevice = &enableBufferDeviceAddress;
    }

    if (m_support.subgroupSizeControl || m_support.computeFullSubgroups) {
      enableSubgroupSizeControl.subgroupSizeControl =
          m_support.subgroupSizeControl ? VK_TRUE : VK_FALSE;
      enableSubgroupSizeControl.computeFullSubgroups =
          m_support.computeFullSubgroups ? VK_TRUE : VK_FALSE;
      enableSubgroupSizeControl.pNext = pNextDevice;
      pNextDevice = &enableSubgroupSizeControl;
    }

    if (m_support.cooperativeMatrix) {
      enableCooperativeMatrix.cooperativeMatrix = VK_TRUE;
      enableCooperativeMatrix.pNext = pNextDevice;
      pNextDevice = &enableCooperativeMatrix;
    }

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = pNextDevice;
    createInfo.flags = 0;
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.queueCreateInfoCount = 1;
    createInfo.ppEnabledExtensionNames = enabledDeviceExtensions.data();
    createInfo.enabledExtensionCount =
        static_cast<uint32_t>(enabledDeviceExtensions.size());
    createInfo.ppEnabledLayerNames = nullptr;
    createInfo.enabledLayerCount = 0;
    createInfo.pEnabledFeatures = &enabledBaseFeatures;

    VkResult result =
        vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to create logical device.");
    }

    vkGetDeviceQueue(m_device, sel.family, 0, &m_queue);
  }

  { // Setup VMA.
    VmaVulkanFunctions vulkanFunctions;
    std::memset(&vulkanFunctions, 0, sizeof(VmaVulkanFunctions));
    vulkanFunctions.vkGetInstanceProcAddr = &vkGetInstanceProcAddr;
    vulkanFunctions.vkGetDeviceProcAddr = &vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo vmaCreateInfo;
    std::memset(&vmaCreateInfo, 0, sizeof(VmaAllocatorCreateInfo));
    vmaCreateInfo.flags = VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT;

    if (m_support.dedicatedAllocation) {
      vmaCreateInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT;
    }
    if (m_support.bindMemory2) {
      vmaCreateInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT;
    }
    if (m_support.maintenance4) {
      vmaCreateInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE4_BIT;
    }
    if (m_support.maintenance5) {
      vmaCreateInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT;
    }
    if (hasMemoryBudgetExt) {
      vmaCreateInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
    }
    if (m_support.bufferDeviceAddress) {
      vmaCreateInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    }
    if (hasMemoryPriorityExt) {
      vmaCreateInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT;
    }
    if (hasAmdDeviceCoherentMemoryExt) {
      vmaCreateInfo.flags |=
          VMA_ALLOCATOR_CREATE_AMD_DEVICE_COHERENT_MEMORY_BIT;
    }
#if defined(_WIN32)
    if (hasExternalMemoryWin32Ext) {
      vmaCreateInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_EXTERNAL_MEMORY_WIN32_BIT;
    }
#endif

    vmaCreateInfo.vulkanApiVersion = m_support.deviceApiVersion;
    vmaCreateInfo.physicalDevice = m_physicalDevice;
    vmaCreateInfo.device = m_device;
    vmaCreateInfo.instance = m_instance;
    vmaCreateInfo.pVulkanFunctions = &vulkanFunctions;

    VkResult result = vmaCreateAllocator(&vmaCreateInfo, &m_vma);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to create Vulkan memory allocator.");
    }
  }
}

Context::~Context() {
  if (m_vma != VK_NULL_HANDLE) {
    vmaDestroyAllocator(m_vma);
    m_vma = VK_NULL_HANDLE;
  }

  if (m_device != VK_NULL_HANDLE) {
    vkDestroyDevice(m_device, nullptr);
    m_device = VK_NULL_HANDLE;
  }

  if (m_debugMessenger != VK_NULL_HANDLE) {
    DestroyDebugUtilsMessengerEXT(m_instance, m_debugMessenger, nullptr);
    m_debugMessenger = VK_NULL_HANDLE;
  }

  if (m_instance != VK_NULL_HANDLE) {
    vkDestroyInstance(m_instance, nullptr);
    m_instance = VK_NULL_HANDLE;
  }
}

Buffer Context::createBuffer(size_t size, VkBufferUsageFlags usage,
                             VmaAllocationCreateFlags flags) {
  VkBufferCreateInfo bufferInfo;
  std::memset(&bufferInfo, 0, sizeof(VkBufferCreateInfo));
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage;

  VmaAllocationCreateInfo allocInfo;
  std::memset(&allocInfo, 0, sizeof(VmaAllocationCreateInfo));
  allocInfo.flags = flags;
  allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

  Buffer buffer;
  VkResult result =
      vmaCreateBuffer(m_vma, &bufferInfo, &allocInfo, &buffer.vkbuffer,
                      &buffer.allocation, nullptr);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create buffer (vulkan error).");
  }
  return buffer;
}
void Context::destroyBuffer(const Buffer &buffer) {
  assert(buffer.vkbuffer != VK_NULL_HANDLE);
  assert(buffer.allocation != VK_NULL_HANDLE);
  vmaDestroyBuffer(m_vma, buffer.vkbuffer, buffer.allocation);
}
VkDescriptorSetLayout Context::createDescriptorSetLayout(
    memory::span<const VkDescriptorSetLayoutBinding> bindings) {
  VkDescriptorSetLayoutCreateInfo layoutInfo;
  std::memset(&layoutInfo, 0, sizeof(VkDescriptorSetLayoutCreateInfo));
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.pBindings = bindings.data();
  layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());

  VkDescriptorSetLayout layout;
  {
    VkResult result =
        vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &layout);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to create descriptor set layout");
    }
  }
  return layout;
}
void Context::destroyDescriptorSetLayout(VkDescriptorSetLayout layout) {
  assert(layout != VK_NULL_HANDLE);
  vkDestroyDescriptorSetLayout(m_device, layout, nullptr);
}

VkPipelineLayout Context::createPipelineLayout(
    memory::span<const VkDescriptorSetLayout> descriptorLayouts,
    uint32_t pushConstantRange) {
  VkPipelineLayoutCreateInfo layoutInfo{};
  std::memset(&layoutInfo, 0, sizeof(VkPipelineLayoutCreateInfo));
  layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorLayouts.size());
  layoutInfo.pSetLayouts = descriptorLayouts.data();
  VkPushConstantRange pushConstant;
  pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pushConstant.size = pushConstantRange;
  pushConstant.offset = 0;
  if (pushConstantRange > 0) {
    layoutInfo.pPushConstantRanges = &pushConstant;
    layoutInfo.pushConstantRangeCount = 1;
  }

  VkPipelineLayout layout;
  {
    VkResult result =
        vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &layout);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to create pipeline layout.");
    }
  }
  return layout;
}
void Context::destroyPipelineLayout(VkPipelineLayout layout) {
  assert(layout != VK_NULL_HANDLE);
  vkDestroyPipelineLayout(m_device, layout, nullptr);
}

VkPipeline Context::createComputePipeline(
    VkPipelineLayout layout, memory::span<const uint32_t> binary,
    const char *entry, std::optional<uint32_t> subgroupSize) {
  assert(layout != VK_NULL_HANDLE);
  assert(!binary.empty());
  assert(entry != nullptr);

  VkShaderModule module = VK_NULL_HANDLE;
  {
    VkShaderModuleCreateInfo shaderInfo{};
    shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderInfo.pCode = binary.data();
    shaderInfo.codeSize = binary.size() * sizeof(uint32_t);

    const VkResult result =
        vkCreateShaderModule(m_device, &shaderInfo, nullptr, &module);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to create shader module.");
    }
  }

  try {
    if (subgroupSize.has_value() && !m_support.subgroupSizeControl) {
      throw std::runtime_error(
          "Requesting a fixed subgroup size, but subgroup size control is not "
          "enabled for this device/context.");
    }

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = layout;

    pipelineInfo.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = module;
    pipelineInfo.stage.pName = entry;

#if defined(VK_VERSION_1_3)
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSizeInfo{};
#elif defined(VK_EXT_subgroup_size_control)
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT subgroupSizeInfo{};
#endif

    if (subgroupSize.has_value()) {
#if defined(VK_VERSION_1_3)
      subgroupSizeInfo.sType =
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
      subgroupSizeInfo.pNext = nullptr;
      subgroupSizeInfo.requiredSubgroupSize = *subgroupSize;

      pipelineInfo.stage.pNext = &subgroupSizeInfo;
      pipelineInfo.stage.flags |=
          VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
#elif defined(VK_EXT_subgroup_size_control)
      subgroupSizeInfo.sType =
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT;
      subgroupSizeInfo.pNext = nullptr;
      subgroupSizeInfo.requiredSubgroupSize = *subgroupSize;

      pipelineInfo.stage.pNext = &subgroupSizeInfo;
      pipelineInfo.stage.flags |=
          VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
#else
      throw std::runtime_error(
          "This build was compiled without subgroup-size-control pipeline "
          "create definitions.");
#endif
    }

    VkPipeline pipeline = VK_NULL_HANDLE;
    const VkResult result = vkCreateComputePipelines(
        m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to create compute pipeline.");
    }

    vkDestroyShaderModule(m_device, module, nullptr);
    return pipeline;
  } catch (...) {
    vkDestroyShaderModule(m_device, module, nullptr);
    throw;
  }
}

void Context::destroyPipeline(VkPipeline pipeline) {
  assert(pipeline != VK_NULL_HANDLE);
  vkDestroyPipeline(m_device, pipeline, nullptr);
}

VkCommandPool Context::createCommandPool() {
  VkCommandPoolCreateInfo poolInfo;
  std::memset(&poolInfo, 0, sizeof(VkCommandPoolCreateInfo));
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = m_queueFamily;
  VkCommandPool pool;
  {
    VkResult result = vkCreateCommandPool(m_device, &poolInfo, nullptr, &pool);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to create command pool.");
    }
  }
  return pool;
}

void Context::destroyCommandPool(VkCommandPool cmdPool) {
  vkDestroyCommandPool(m_device, cmdPool, nullptr);
}

VkCommandBuffer Context::allocCommandBuffer(VkCommandPool cmdPool) {
  VkCommandBufferAllocateInfo allocInfo;
  std::memset(&allocInfo, 0, sizeof(VkCommandBufferAllocateInfo));
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = cmdPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;
  VkCommandBuffer cmd;
  {
    VkResult result = vkAllocateCommandBuffers(m_device, &allocInfo, &cmd);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to allocate command buffer.");
    }
  }
  return cmd;
}

void Context::freeCommandBuffer(VkCommandPool cmdPool, VkCommandBuffer cmd) {
  vkFreeCommandBuffers(m_device, cmdPool, 1, &cmd);
}

void Context::beginCommandBuffer(VkCommandBuffer cmd) {
  VkCommandBufferBeginInfo beginInfo;
  std::memset(&beginInfo, 0, sizeof(VkCommandBufferBeginInfo));
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  {
    VkResult result = vkBeginCommandBuffer(cmd, &beginInfo);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to begin command buffer.");
    }
  }
}

void Context::endCommandBuffer(VkCommandBuffer cmd) { vkEndCommandBuffer(cmd); }

void Context::submit(VkCommandBuffer cmd) {
  VkSubmitInfo submitInfo;
  std::memset(&submitInfo, 0, sizeof(VkSubmitInfo));
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmd;
  submitInfo.pSignalSemaphores = nullptr;
  submitInfo.signalSemaphoreCount = 0;
  submitInfo.pWaitSemaphores = nullptr;
  submitInfo.waitSemaphoreCount = 0;
  submitInfo.pWaitDstStageMask = nullptr;

  {
    VkResult result = vkQueueSubmit(m_queue, 1, &submitInfo, nullptr);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to submit to queue.");
    }
  }
}

void Context::waitIdle() {
  VkResult result = vkQueueWaitIdle(m_queue);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to wait for queue idle.");
  }
}

VkCommandBuffer Context::allocBeginCommandBuffer(VkCommandPool cmdPool) {
  VkCommandBuffer cmd = allocCommandBuffer(cmdPool);
  beginCommandBuffer(cmd);
  return cmd;
}

void Context::endSubmitWaitCommandBuffer(VkCommandPool cmdPool,
                                         VkCommandBuffer cmd) {
  endCommandBuffer(cmd);
  submit(cmd);
  waitIdle();
  freeCommandBuffer(cmdPool, cmd);
}

VkDescriptorPool
Context::createDescriptorPool(uint32_t maxSets,
                              memory::span<const VkDescriptorPoolSize> sizes) {
  VkDescriptorPoolCreateInfo poolInfo;
  std::memset(&poolInfo, 0, sizeof(VkDescriptorPoolCreateInfo));
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.maxSets = maxSets;
  poolInfo.poolSizeCount = static_cast<uint32_t>(sizes.size());
  poolInfo.pPoolSizes = sizes.data();

  VkDescriptorPool pool;
  VkResult result = vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &pool);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create descriptor pool");
  }
  return pool;
}

void Context::destroyDescriptorPool(VkDescriptorPool pool) {
  vkDestroyDescriptorPool(m_device, pool, nullptr);
}

VkDescriptorSet Context::allocDescriptorSet(VkDescriptorPool pool,
                                            VkDescriptorSetLayout layout) {
  VkDescriptorSetAllocateInfo allocInfo;
  std::memset(&allocInfo, 0, sizeof(VkDescriptorSetAllocateInfo));
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = pool;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &layout;
  VkDescriptorSet set;
  VkResult result = vkAllocateDescriptorSets(m_device, &allocInfo, &set);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create descriptor set");
  }
  return set;
}

void Context::allocDescriptorSets(
    VkDescriptorPool pool, memory::span<const VkDescriptorSetLayout> layouts,
    VkDescriptorSet *sets) {
  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = pool;
  allocInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
  allocInfo.pSetLayouts = layouts.data();
  VkResult result = vkAllocateDescriptorSets(m_device, &allocInfo, sets);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create descriptor set");
  }
}

void Context::updateDescriptorSets(
    memory::span<const VkWriteDescriptorSet> writeInfos) {
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeInfos.size()),
                         writeInfos.data(), 0, nullptr);
}

void Context::copy(VmaAllocation dst, const void *src, size_t size) {
  VkResult result = vmaCopyMemoryToAllocation(m_vma, src, dst, 0, size);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to copy memory to allocation.");
  }
}

void Context::copy(void *dst, VmaAllocation src, size_t size, size_t offset) {
  VkResult result = vmaCopyAllocationToMemory(m_vma, src, offset, dst, size);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to copy memory from allocation.");
  }
}

void Context::cmdCopy(VkCommandBuffer cmd, Buffer dst, Buffer src, size_t size,
                      size_t dstOffset, size_t srcOffset) {
  VkBufferCopy copy;
  copy.size = size;
  copy.srcOffset = srcOffset;
  copy.dstOffset = dstOffset;
  vkCmdCopyBuffer(cmd, src.vkbuffer, dst.vkbuffer, 1, &copy);
}

void Context::cmdMemoryBarrier(VkCommandBuffer cmd,
                               VkPipelineStageFlags srcStage,
                               VkPipelineStageFlags dstStage,
                               VkAccessFlags srcAccess,
                               VkAccessFlags dstAccess) {
  VkMemoryBarrier memoryBarrier;
  std::memset(&memoryBarrier, 0, sizeof(VkMemoryBarrier));
  memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  memoryBarrier.srcAccessMask = srcAccess;
  memoryBarrier.dstAccessMask = dstAccess;

  vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 1, &memoryBarrier, 0,
                       nullptr, 0, nullptr);
}

void Context::cmdMemoryBarrierComputeShader(VkCommandBuffer cmd) {
  cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                   VK_ACCESS_SHADER_READ_BIT,
                   VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

void Context::cmdBufferBarrier(VkCommandBuffer cmd, Buffer buffer,
                               VkPipelineStageFlags srcStage,
                               VkPipelineStageFlags dstStage,
                               VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                               VkDeviceSize offset, VkDeviceSize size) {
  VkBufferMemoryBarrier bufferBarrier;
  bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  bufferBarrier.pNext = nullptr;
  bufferBarrier.srcAccessMask = srcAccess;
  bufferBarrier.dstAccessMask = dstAccess;
  bufferBarrier.srcQueueFamilyIndex = m_queueFamily;
  bufferBarrier.dstQueueFamilyIndex = m_queueFamily;
  bufferBarrier.buffer = buffer.vkbuffer;
  bufferBarrier.offset = offset;
  bufferBarrier.size = size;

  vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 1,
                       &bufferBarrier, 0, nullptr);
}
} // namespace denox::runtime
