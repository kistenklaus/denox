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
    features.robustBufferAccess = VK_FALSE;
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

Buffer Context::createBuffer(std::size_t size, VkBufferUsageFlags usage,
                             VmaAllocationCreateFlags flags) {
  VkBufferCreateInfo bufferInfo;
  std::memset(&bufferInfo, 0, sizeof(VkBufferCreateInfo));
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage;

  VmaAllocationCreateInfo allocInfo = {};
  allocInfo.flags = flags;
  allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

  Buffer buffer;
  VkResult result =
      vmaCreateBuffer(m_vma, &bufferInfo, &allocInfo, &buffer.buffer,
                      &buffer.allocation, nullptr);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create buffer.");
  }
  return buffer;
}
void Context::destroyBuffer(Buffer buffer) {
  assert(buffer.buffer != VK_NULL_HANDLE);
  assert(buffer.allocation != VK_NULL_HANDLE);
  vmaDestroyBuffer(m_vma, buffer.buffer, buffer.allocation);
}
VkDescriptorSetLayout Context::createDescriptorSetLayout(
    std::span<const VkDescriptorSetLayoutBinding> bindings) {
  VkDescriptorSetLayoutCreateInfo layoutInfo;
  std::memset(&layoutInfo, 0, sizeof(VkDescriptorSetLayoutCreateInfo));
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.pBindings = bindings.data();
  layoutInfo.bindingCount = bindings.size();

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
    std::span<const VkDescriptorSetLayout> descriptorLayouts,
    std::uint32_t pushConstantRange) {
  VkPipelineLayoutCreateInfo layoutInfo;
  std::memset(&layoutInfo, 0, sizeof(VkPipelineLayoutCreateInfo));
  layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layoutInfo.setLayoutCount = descriptorLayouts.size();
  layoutInfo.pSetLayouts = descriptorLayouts.data();
  VkPushConstantRange pushConstant;
  pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pushConstant.size = pushConstantRange;
  pushConstant.offset = 0;
  layoutInfo.pPushConstantRanges = &pushConstant;
  layoutInfo.pushConstantRangeCount = 1;

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
                                          std::span<const std::uint32_t> binary,
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
    shaderInfo.codeSize = binary.size() * sizeof(std::uint32_t);
    VkResult result =
        vkCreateShaderModule(m_device, &shaderInfo, nullptr, &module);
    if (result != VK_SUCCESS) {
      throw std::runtime_error("Failed to create shader module.");
    }
  }

  VkComputePipelineCreateInfo pipelineInfo;
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
} // namespace denox::runtime
