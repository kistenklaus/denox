#include "context.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <fmt/format.h>
#include <fmt/printf.h>
#include <stdexcept>
#include <type_traits>
#include <vector>
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

static std::pair<VkInstance, VkDebugUtilsMessengerEXT> createInstance() {
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
  VkInstance instance;
  VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create vulkan instance.");
  }

  VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
  if (validationLayerSupported) {
    CreateDebugUtilsMessengerEXT(instance, &debugUtilsMessengerCreateInfo,
                                 nullptr, &debugMessenger);
  }

  return {instance, debugMessenger};
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

static VkPhysicalDevice selectPhysicalDevice(VkInstance instance,
                                             const char *deviceName) {

  std::uint32_t deviceCount;
  vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
  std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
  vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.data());

  if (physicalDevices.empty()) {
    throw std::runtime_error(
        "Failed to select physical device: No Vulkan device found.");
  }

  if (deviceName == nullptr) {
    for (VkPhysicalDevice d : physicalDevices) {
      VkPhysicalDeviceProperties props;
      vkGetPhysicalDeviceProperties(d, &props);
      if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        return d;
      }
    }
    return physicalDevices[0];
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
    throw std::runtime_error(
        fmt::format("Failed to select physical device: pattern \"{}\" did not "
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
  return matches.front();
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

static std::tuple<VkDevice, std::uint32_t, VkQueue>
createDevice(VkInstance instance, VkPhysicalDevice physicalDevice,
             bool validation) {
  ComputeQueueSelection sel = pick_best_compute_queue_family(physicalDevice);
  VkDeviceQueueCreateInfo queueCreateInfo;
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCreateInfo.pNext = nullptr;
  queueCreateInfo.flags = 0;
  float queuePriority = 1.0f;
  queueCreateInfo.pQueuePriorities = &queuePriority;
  queueCreateInfo.queueCount = 1;
  queueCreateInfo.queueFamilyIndex = sel.family;

  VkPhysicalDeviceFeatures features;
  std::memset(&features, 0, sizeof(VkPhysicalDeviceFeatures));

  std::vector<const char *> layers;
  std::vector<const char *> extentions;
#ifdef __APPLE__
  extentions.push_back("VK_KHR_portability_subset");
#endif

  VkDeviceCreateInfo createInfo;
  createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  createInfo.pNext = nullptr;
  createInfo.flags = 0;
  createInfo.pQueueCreateInfos = &queueCreateInfo;
  createInfo.queueCreateInfoCount = 1;
  createInfo.ppEnabledExtensionNames = extentions.data();
  createInfo.enabledExtensionCount = extentions.size();
  createInfo.ppEnabledLayerNames = layers.data();
  createInfo.enabledLayerCount = layers.size();
  createInfo.pEnabledFeatures = &features;
  VkDevice device;
  VkResult result =
      vkCreateDevice(physicalDevice, &createInfo, nullptr, &device);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create logical device");
  }
  VkQueue queue;
  vkGetDeviceQueue(device, sel.family, 0, &queue);
  return {device, sel.family, queue};
}

Context::Context(const char *deviceName)
    : m_instance(VK_NULL_HANDLE), m_device(VK_NULL_HANDLE),
      m_physicalDevice(VK_NULL_HANDLE), m_queue(VK_NULL_HANDLE) {
  auto [instance, debugMessenger] = createInstance();
  m_instance = instance;
  m_debugMessenger = debugMessenger;
  m_physicalDevice = selectPhysicalDevice(m_instance, deviceName);
  auto [device, queueFamily, queue] = createDevice(m_instance, m_physicalDevice,
                                      m_debugMessenger != VK_NULL_HANDLE);
  m_device = device;
  m_queueFamily = queueFamily;
  m_queue = queue;
}

Context::~Context() {

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
