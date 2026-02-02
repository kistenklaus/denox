#include "context.hpp"
#include "vk_mem_alloc.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <fmt/format.h>
#include <fmt/printf.h>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>

namespace denox::runtime {

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
  return VK_FALSE;
  switch (severity) {
  case Severity::None:
    return VK_FALSE;
  case Severity::Verbose:
#ifndef DENOX_QUIET
    fmt::println("\x1B[37m[Validation-Layer]:\x1B[0m {}",
                 pCallbackData->pMessage);
#endif
    break;
  case Severity::Info:
#ifndef DENOX_QUIET
    fmt::println("\x1B[34m[Validation-Layer]:\x1B[0m {}",
                 pCallbackData->pMessage);
#endif
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

VkResult GetPhysicalDeviceCooperativeMatrixPropertiesKHR(
    VkInstance instance, VkPhysicalDevice physicalDevice, uint32_t *count,
    VkCooperativeMatrixPropertiesKHR *properties) {

  auto func =
      reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
          vkGetInstanceProcAddr(
              instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
  if (func != nullptr) {
    return func(physicalDevice, count, properties);
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

Context::Context(const char *deviceName, ApiVersion target_env)
    : m_instance(VK_NULL_HANDLE), m_device(VK_NULL_HANDLE),
      m_physicalDevice(VK_NULL_HANDLE), m_queue(VK_NULL_HANDLE) {

  uint32_t vulkanApiVersion;
  { // Create instance.
    VkApplicationInfo appInfo;
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pNext = nullptr;

#ifdef VK_API_VERSION_1_4
#define VK_API_VERSION_LATEST VK_API_VERSION_1_4;
#elif defined(VK_API_VERSION_1_3)
#define VK_API_VERSION_LATEST VK_API_VERSION_1_3;
#else
#define VK_API_VERSION_LATEST VK_API_VERSION_1_1;
#endif

    switch (target_env) {
    case ApiVersion::VULKAN_1_0:
#ifdef VK_API_VERSION_1_0
      appInfo.apiVersion = VK_API_VERSION_1_0;
#else
      appInfo.apiVersion = VK_API_VERSION_LATEST;
#endif
      break;
    case ApiVersion::VULKAN_1_1:
#ifdef VK_API_VERSION_1_1
      appInfo.apiVersion = VK_API_VERSION_1_1;
#else
      appInfo.apiVersion = VK_API_VERSION_LATEST;
#endif
      break;
    case ApiVersion::VULKAN_1_2:
#ifdef VK_API_VERSION_1_2
      appInfo.apiVersion = VK_API_VERSION_1_2;
#else
      appInfo.apiVersion = VK_API_VERSION_LATEST;
#endif
      break;
    case ApiVersion::VULKAN_1_3:
#ifdef VK_API_VERSION_1_3
      appInfo.apiVersion = VK_API_VERSION_1_3;
#else
      appInfo.apiVersion = VK_API_VERSION_LATEST;
#endif
      break;
    case ApiVersion::VULKAN_1_4:
#ifdef VK_API_VERSION_1_4
      appInfo.apiVersion = VK_API_VERSION_1_4;
#else
      appInfo.apiVersion = VK_API_VERSION_LATEST;
#endif
      break;
    }

    vulkanApiVersion = appInfo.apiVersion;
    appInfo.applicationVersion = DENOX_VERSION;
    appInfo.pApplicationName = "denox-runtime";
    appInfo.engineVersion = DENOX_VERSION;
    appInfo.pEngineName = "denox";
    VkInstanceCreateInfo createInfo;
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pNext = nullptr;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.flags = 0;

    memory::vector<const char *> extentions;
    memory::vector<const char *> layers;
#ifdef __APPLE__
    extentions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
    const bool validationLayerSupported =
        checkLayerSupport("VK_LAYER_KHRONOS_validation");
    if (validationLayerSupported) {
      layers.push_back("VK_LAYER_KHRONOS_validation");
    }
    VkDebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfo{};
    if (validationLayerSupported) {
      extentions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
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
      debugUtilsMessengerCreateInfo.pUserData = nullptr; // Optional
      createInfo.pNext = &debugUtilsMessengerCreateInfo;
    }

    createInfo.ppEnabledExtensionNames = extentions.data();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extentions.size());
    createInfo.ppEnabledLayerNames = layers.data();
    createInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
    VkResult result = vkCreateInstance(&createInfo, nullptr, &m_instance);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to create vulkan instance.");
    }

    if (validationLayerSupported) {
      CreateDebugUtilsMessengerEXT(m_instance, &debugUtilsMessengerCreateInfo,
                                   nullptr, &m_debugMessenger);
    }
  }
  { // Select physical device.
    do {
      uint32_t deviceCount;
      vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);
      memory::vector<VkPhysicalDevice> physicalDevices(deviceCount);
      vkEnumeratePhysicalDevices(m_instance, &deviceCount,
                                 physicalDevices.data());

      if (physicalDevices.empty()) {
        throw std::runtime_error(
            "Failed to select physical device: No Vulkan device found.");
      }

      if (deviceName == nullptr) {
        for (VkPhysicalDevice d : physicalDevices) {
          VkPhysicalDeviceProperties props;
          vkGetPhysicalDeviceProperties(d, &props);
          if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            m_physicalDevice = d;
            break;
          }
        }
        m_physicalDevice = physicalDevices[0];
        break;
      }

      std::string devicePattern(deviceName);

      memory::vector<VkPhysicalDevice> matches;
      for (VkPhysicalDevice d : physicalDevices) {
        VkPhysicalDeviceProperties props;
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
          VkPhysicalDeviceProperties props;
          vkGetPhysicalDeviceProperties(d, &props);
          list += std::string(props.deviceName) + "; ";
        }
        throw std::runtime_error(
            fmt::format("Failed to select physical device: pattern \"{}\" is "
                        "ambiguous, matches multiple devices: {}",
                        devicePattern, list));
      }
      assert(matches.size() == 1);
      m_physicalDevice = matches.front();
    } while (false);
  }
  {
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(m_physicalDevice, &properties);
    m_timestampPeriod = properties.limits.timestampPeriod;
  }

  VkPhysicalDeviceFeatures features;
  {
    std::memset(&features, 0, sizeof(VkPhysicalDeviceFeatures));
    vkGetPhysicalDeviceFeatures(m_physicalDevice, &features);
    std::memset(&features, 0, sizeof(VkPhysicalDeviceFeatures));
    features.robustBufferAccess = VK_TRUE;
    features.shaderInt16 = VK_TRUE;
  }

  memory::vector<const char *> layers;
  memory::vector<const char *> extentions;

  void *pNextDevice = nullptr;
#ifdef VK_API_VERSION_1_1
  VkPhysicalDeviceVulkan11Features features11;
  {
    std::memset(&features11, 0, sizeof(VkPhysicalDeviceVulkan11Features));
    features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    VkPhysicalDeviceFeatures2 features2;
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &features11;
    vkGetPhysicalDeviceFeatures2(m_physicalDevice, &features2);

    std::memset(&features11, 0, sizeof(VkPhysicalDeviceVulkan11Features));
    features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    features11.storageBuffer16BitAccess = VK_TRUE;

    features11.pNext = pNextDevice;
    pNextDevice = &features11;
  }
#endif
#ifdef VK_API_VERSION_1_2
  VkPhysicalDeviceVulkan12Features features12;
  {
    std::memset(&features12, 0, sizeof(VkPhysicalDeviceVulkan12Features));
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    VkPhysicalDeviceFeatures2 features2;
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &features12;
    vkGetPhysicalDeviceFeatures2(m_physicalDevice, &features2);

    std::memset(&features12, 0, sizeof(VkPhysicalDeviceVulkan12Features));
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.pNext = pNextDevice;
    features12.vulkanMemoryModel = VK_TRUE;
    features12.shaderFloat16 = VK_TRUE;
    pNextDevice = &features12;
  }
#endif
#ifdef VK_API_VERSION_1_3
  VkPhysicalDeviceVulkan13Features features13;
  {
    std::memset(&features13, 0, sizeof(VkPhysicalDeviceVulkan13Features));
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    VkPhysicalDeviceFeatures2 features2;
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &features13;
    vkGetPhysicalDeviceFeatures2(m_physicalDevice, &features2);
    std::memset(&features13, 0, sizeof(VkPhysicalDeviceVulkan13Features));
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features13.pNext = pNextDevice;
    pNextDevice = &features13;
  }
#endif
#ifdef VK_API_VERSION_1_4
  VkPhysicalDeviceVulkan14Features features14;
  {
    std::memset(&features14, 0, sizeof(VkPhysicalDeviceVulkan14Features));
    features14.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
    VkPhysicalDeviceFeatures2 features2;
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &features14;
    vkGetPhysicalDeviceFeatures2(m_physicalDevice, &features2);

    std::memset(&features14, 0, sizeof(VkPhysicalDeviceVulkan14Features));
    features14.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
    // features14.globalPriorityQuery = true;
    // features14.shaderSubgroupRotate = true;
    // features14.shaderSubgroupRotateClustered = true;
    // features14.shaderFloatControls2 = true;
    // features14.shaderExpectAssume = true;
    // features14.rectangularLines = true;
    // features14.bresenhamLines = true;
    // features14.smoothLines = true;
    // features14.stippledRectangularLines = true;
    // features14.stippledBresenhamLines = true;
    // features14.vertexAttributeInstanceRateDivisor = true;
    // features14.vertexAttributeInstanceRateZeroDivisor = true;
    // features14.indexTypeUint8 = true;
    // features14.dynamicRenderingLocalRead = true;
    // features14.pipelineProtectedAccess = true;
    features14.pipelineRobustness =
        false; // <- very interessting for finding OOB accecsses.
    // features14.hostImageCopy = true;
    // features14.pushDescriptor = true;
    // features14.maintenance5 = true;
    // features14.maintenance6 = true;

    features14.pNext = pNextDevice;
    pNextDevice = &features14;
  }
#endif
#ifdef VK_KHR_cooperative_matrix
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopmatFeatures;
  {
    std::memset(&coopmatFeatures, 0,
                sizeof(VkPhysicalDeviceCooperativeMatrixFeaturesKHR));
    coopmatFeatures.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    VkPhysicalDeviceFeatures2 features2;
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &coopmatFeatures;
    vkGetPhysicalDeviceFeatures2(m_physicalDevice, &features2);
    if (coopmatFeatures.cooperativeMatrix == VK_TRUE) {
      extentions.push_back("VK_KHR_cooperative_matrix");
      coopmatFeatures.pNext = pNextDevice;
      pNextDevice = &coopmatFeatures;
    }
  }
#endif

  bool dedicatedAllocation = false;
  bool bindMemory2 = false;
  bool maintenance4 = false;
  bool maintenance5 = false;
  bool memoryBudget = false;
  bool bufferDeviceAddress = false;
  bool memoryPriority = false;
  bool amdDeviceCoherentMemory = false;
  bool externalMemoryWin32 = false;

    
  { // Create logical device and compute queue.
    ComputeQueueSelection sel =
        pick_best_compute_queue_family(m_physicalDevice);
    m_queueFamily = sel.family;
    VkDeviceQueueCreateInfo queueCreateInfo;
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.pNext = nullptr;
    queueCreateInfo.flags = 0;
    float queuePriority = 1.0f;
    queueCreateInfo.pQueuePriorities = &queuePriority;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.queueFamilyIndex = sel.family;

#ifdef __APPLE__
    extentions.push_back("VK_KHR_portability_subset");
#endif
    { // Query extention support.
      std::uint32_t ecount = 0;
      vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &ecount,
                                           nullptr);
      std::vector<VkExtensionProperties> supported(ecount);
      vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &ecount,
                                           supported.data());
      for (const VkExtensionProperties &ext : supported) {

#if defined(VK_EXT_pci_bus_info) && VK_EXT_pci_bus_info == 1
        if (std::strcmp(ext.extensionName,
                        VK_EXT_PCI_BUS_INFO_EXTENSION_NAME) == 0) {
          extentions.push_back(VK_EXT_PCI_BUS_INFO_EXTENSION_NAME);
          m_extPCIbufInfoAvailable = true;
        }
#endif
      }
    }

    VkDeviceCreateInfo createInfo;
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = pNextDevice;
    createInfo.flags = 0;
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.queueCreateInfoCount = 1;
    createInfo.ppEnabledExtensionNames = extentions.data();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extentions.size());
    createInfo.ppEnabledLayerNames = layers.data();
    createInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
    createInfo.pEnabledFeatures = &features;
    VkResult result =
        vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to create logical device");
    }
    vkGetDeviceQueue(m_device, sel.family, 0, &m_queue);
  }
  { // Setup vma.
    VmaVulkanFunctions vulkanFunctions;
    std::memset(&vulkanFunctions, 0, sizeof(VmaVulkanFunctions));
    vulkanFunctions.vkGetInstanceProcAddr = &vkGetInstanceProcAddr;
    vulkanFunctions.vkGetDeviceProcAddr = &vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo vmaCreateInfo;
    std::memset(&vmaCreateInfo, 0, sizeof(VmaAllocatorCreateInfo));
    vmaCreateInfo.flags = VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT;
    if (dedicatedAllocation) {
      vmaCreateInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT;
    }
    if (bindMemory2) {
      vmaCreateInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT;
    }
    if (maintenance4) {
      vmaCreateInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE4_BIT;
    }
    if (maintenance5) {
      vmaCreateInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT;
    }
    if (memoryBudget) {
      vmaCreateInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
    }
    if (bufferDeviceAddress) {
      vmaCreateInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    }
    if (memoryPriority) {
      vmaCreateInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT;
    }
    if (amdDeviceCoherentMemory) {
      vmaCreateInfo.flags |=
          VMA_ALLOCATOR_CREATE_AMD_DEVICE_COHERENT_MEMORY_BIT;
    }
    if (externalMemoryWin32) {
      vmaCreateInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_EXTERNAL_MEMORY_WIN32_BIT;
    }
    vmaCreateInfo.vulkanApiVersion = vulkanApiVersion;
    vmaCreateInfo.physicalDevice = m_physicalDevice;
    vmaCreateInfo.device = m_device;
    vmaCreateInfo.instance = m_instance;
    vmaCreateInfo.pVulkanFunctions = &vulkanFunctions;

    VkResult result = vmaCreateAllocator(&vmaCreateInfo, &m_vma);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to create vulkan memory allocator");
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
VkPipeline Context::createComputePipeline(VkPipelineLayout layout,
                                          memory::span<const uint32_t> binary,
                                          const char *entry) {
  assert(layout != nullptr);
  assert(!binary.empty());
  assert(entry != nullptr);
  VkShaderModule module;
  {
    VkShaderModuleCreateInfo shaderInfo;
    std::memset(&shaderInfo, 0, sizeof(VkShaderModuleCreateInfo));
    shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderInfo.pCode = binary.data();
    shaderInfo.codeSize = binary.size() * sizeof(uint32_t);
    VkResult result =
        vkCreateShaderModule(m_device, &shaderInfo, nullptr, &module);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to create shader module.");
    }
  }

  VkComputePipelineCreateInfo pipelineInfo{};
  std::memset(&pipelineInfo, 0, sizeof(VkComputePipelineCreateInfo));
  pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;

  pipelineInfo.stage.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipelineInfo.stage.module = module;
  pipelineInfo.stage.pName = entry;

  pipelineInfo.layout = layout;

  VkPipeline pipeline;
  {
    VkResult result = vkCreateComputePipelines(
        m_device, nullptr, 1, &pipelineInfo, nullptr, &pipeline);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to create compute pipeline.");
    }
  }
  vkDestroyShaderModule(m_device, module, nullptr);
  return pipeline;
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
