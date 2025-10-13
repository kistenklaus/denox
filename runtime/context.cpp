#include "context.hpp"
#include "vk_mem_alloc.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <fmt/format.h>
#include <fmt/printf.h>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <vma.hpp>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>

namespace denox::runtime {

static VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageType,
              const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
              void *pUserData) {
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
  switch (severity) {
  case Severity::None:
    return VK_FALSE;
  case Severity::Verbose:
    fmt::println("\x1B[37m[Validation-Layer]:\x1B[0m\n{}",
                 pCallbackData->pMessage);
    break;
  case Severity::Info:
    fmt::println("\x1B[34m[Validation-Layer]:\x1B[0m\n{}",
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
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks *pAllocator) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) {
    func(instance, debugMessenger, pAllocator);
  }
}

static bool checkLayerSupport(const char *layerName) {
  uint32_t layerCount;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
  std::vector<VkLayerProperties> layers(layerCount);
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
  std::vector<VkQueueFamilyProperties> props(count);
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

Context::Context(const char *deviceName)
    : m_instance(VK_NULL_HANDLE), m_device(VK_NULL_HANDLE),
      m_physicalDevice(VK_NULL_HANDLE), m_queue(VK_NULL_HANDLE) {

  std::uint32_t vulkanApiVersion;
  { // Create instance.
    VkApplicationInfo appInfo;
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pNext = nullptr;
#ifdef VK_API_VERSION_1_4
    appInfo.apiVersion = VK_API_VERSION_1_4;
#elif defined(VK_API_VERSION_1_3)
    appInfo.apiVersion = VK_API_VERSION_1_3;
#else
    appInfo.apiVersion = VK_API_VERSION_1_1;
#endif
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

    std::vector<const char *> extentions;
    std::vector<const char *> layers;
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
          // VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
          // VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
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
    createInfo.enabledExtensionCount = extentions.size();
    createInfo.ppEnabledLayerNames = layers.data();
    createInfo.enabledLayerCount = layers.size();
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
      std::uint32_t deviceCount;
      vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);
      std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
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

      std::vector<VkPhysicalDevice> matches;
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

  VkPhysicalDeviceFeatures features;
  {
    std::memset(&features, 0, sizeof(VkPhysicalDeviceFeatures));
    vkGetPhysicalDeviceFeatures(m_physicalDevice, &features);
  }

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
    features12.pNext = pNextDevice;
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
    features14.pNext = pNextDevice;
    pNextDevice = &features14;
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
  std::uint32_t queueFamily;
  { // Create logical device and compute queue.
    ComputeQueueSelection sel =
        pick_best_compute_queue_family(m_physicalDevice);
    queueFamily = sel.family;
    VkDeviceQueueCreateInfo queueCreateInfo;
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.pNext = nullptr;
    queueCreateInfo.flags = 0;
    float queuePriority = 1.0f;
    queueCreateInfo.pQueuePriorities = &queuePriority;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.queueFamilyIndex = sel.family;

    std::vector<const char *> layers;
    std::vector<const char *> extentions;
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
        if (std::strcmp(ext.extensionName, "VK_KHR_dedicated_allocation") ==
            0) {
          extentions.push_back("VK_KHR_dedicated_allocation");
          dedicatedAllocation = true;
        }
        if (std::strcmp(ext.extensionName, "VK_KHR_bind_memory2") == 0) {
          extentions.push_back("VK_KHR_bind_memory2");
          bindMemory2 = true;
        }
        if (std::strcmp(ext.extensionName, "VK_KHR_maintenance4") == 0) {
          extentions.push_back("VK_KHR_maintenance4");
          maintenance4 = true;
        }
        if (std::strcmp(ext.extensionName, "VK_KHR_maintenance5") == 0) {
          extentions.push_back("VK_KHR_maintenance5");
          maintenance5 = true;
        }
        if (std::strcmp(ext.extensionName, "VK_EXT_memory_budget") == 0) {
          extentions.push_back("VK_EXT_memory_budget");
          memoryBudget = true;
        }
        if (std::strcmp(ext.extensionName, "VK_KHR_buffer_device_address") ==
            0) {
          // NOTE: Maybe in some day in the future it would be nice to play
          // around with this because it can probably avoid the overhead of
          // switching pipeline layouts between dispatches as well as improve
          // descriptor update perf.
          // extentions.push_back("VK_KHR_buffer_device_address");
          // bufferDeviceAddress = true;
        }
        if (std::strcmp(ext.extensionName, "VK_EXT_memory_priority") == 0) {
          extentions.push_back("VK_EXT_memory_priority");
          memoryPriority = true;
        }
        if (std::strcmp(ext.extensionName, "VK_AMD_device_coherent_memory") ==
            0) {
          extentions.push_back("VK_AMD_device_coherent_memory");
          amdDeviceCoherentMemory = true;
        }
        if (std::strcmp(ext.extensionName, "VK_KHR_external_memory_win32") ==
            0) {
          extentions.push_back("VK_KHR_external_memory_win32");
          externalMemoryWin32 = true;
        }
      }
    }

    VkDeviceCreateInfo createInfo;
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = pNextDevice;
    createInfo.flags = 0;
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.queueCreateInfoCount = 1;
    createInfo.ppEnabledExtensionNames = extentions.data();
    createInfo.enabledExtensionCount = extentions.size();
    createInfo.ppEnabledLayerNames = layers.data();
    createInfo.enabledLayerCount = layers.size();
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

} // namespace denox::runtime
