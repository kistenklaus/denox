#include "denox/device_info/query/query_subgroup_properties.hpp"
#include "denox/device_info/SubgroupProperties.hpp"
#include <vulkan/vulkan_core.h>

namespace denox {

SubgroupProperties query_subgroup_properties(vk::Instance instance,
                                             vk::PhysicalDevice physicalDevice,
                                             ApiVersion apiVersion) {
  SubgroupProperties out{};
  out.subgroupSize = 0;
  out.supportsBasicOps = false;
  out.supportsVoteOps = false;
  out.supportsArithmeticOps = false;
  out.supportsBallotOps = false;
  out.supportsShuffleOps = false;
  out.supportsShuffleRelativeOps = false;
  out.controlProperties.maxComputeWorkgroupSubgroups = 0;
  out.controlProperties.supported = false;

  auto hasDeviceExtension = [&](const char *name,
                                uint32_t *specVersion = nullptr) -> bool {
    for (const auto &ext :
         physicalDevice.enumerateDeviceExtensionProperties()) {
      if (std::strcmp(ext.extensionName, name) == 0) {
        if (specVersion) {
          *specVersion = ext.specVersion;
        }
        return true;
      }
    }
    return false;
  };

  auto fpGetProperties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
      instance.getProcAddr("vkGetPhysicalDeviceProperties2"));
  auto fpGetProperties2KHR =
      reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2KHR>(
          instance.getProcAddr("vkGetPhysicalDeviceProperties2KHR"));
  auto fpGetFeatures2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
      instance.getProcAddr("vkGetPhysicalDeviceFeatures2"));
  auto fpGetFeatures2KHR =
      reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2KHR>(
          instance.getProcAddr("vkGetPhysicalDeviceFeatures2KHR"));

  if (!fpGetProperties2 && !fpGetProperties2KHR) {
    return out;
  }

  const bool apiAtLeast11 = apiVersion >= ApiVersion::VULKAN_1_1;
  const bool apiAtLeast13 = apiVersion >= ApiVersion::VULKAN_1_3;

  uint32_t subgroupSizeControlExtVersion = 0;
  const bool hasSubgroupSizeControlExt =
      hasDeviceExtension(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
                         &subgroupSizeControlExtVersion);

  const bool canQuerySubgroupProps = apiAtLeast11;
  const bool canQuerySizeControlProps =
      apiAtLeast13 || hasSubgroupSizeControlExt;

  vk::PhysicalDeviceProperties2 props2{};
  vk::PhysicalDeviceSubgroupProperties subgroupProps{};
  vk::PhysicalDeviceSubgroupSizeControlProperties sizeControlProps{};

  void **propsTail = &props2.pNext;
  if (canQuerySubgroupProps) {
    *propsTail = &subgroupProps;
    propsTail = &subgroupProps.pNext;
  }
  if (canQuerySizeControlProps) {
    *propsTail = &sizeControlProps;
    propsTail = &sizeControlProps.pNext;
  }
  *propsTail = nullptr;

  if (fpGetProperties2) {
    fpGetProperties2(static_cast<VkPhysicalDevice>(physicalDevice),
                     reinterpret_cast<VkPhysicalDeviceProperties2 *>(&props2));
  } else {
    fpGetProperties2KHR(
        static_cast<VkPhysicalDevice>(physicalDevice),
        reinterpret_cast<VkPhysicalDeviceProperties2 *>(&props2));
  }

  if (canQuerySubgroupProps) {
    out.subgroupSize = subgroupProps.subgroupSize;

    if (static_cast<bool>(subgroupProps.supportedStages &
                          vk::ShaderStageFlagBits::eCompute)) {
      out.supportsBasicOps =
          static_cast<bool>(subgroupProps.supportedOperations &
                            vk::SubgroupFeatureFlagBits::eBasic);
      out.supportsVoteOps =
          static_cast<bool>(subgroupProps.supportedOperations &
                            vk::SubgroupFeatureFlagBits::eVote);
      out.supportsArithmeticOps =
          static_cast<bool>(subgroupProps.supportedOperations &
                            vk::SubgroupFeatureFlagBits::eArithmetic);
      out.supportsBallotOps =
          static_cast<bool>(subgroupProps.supportedOperations &
                            vk::SubgroupFeatureFlagBits::eBallot);
      out.supportsShuffleOps =
          static_cast<bool>(subgroupProps.supportedOperations &
                            vk::SubgroupFeatureFlagBits::eShuffle);
      out.supportsShuffleRelativeOps =
          static_cast<bool>(subgroupProps.supportedOperations &
                            vk::SubgroupFeatureFlagBits::eShuffleRelative);
    }
  }

  bool subgroupSizeControlFeature = false;

  if (apiAtLeast13 ||
      (hasSubgroupSizeControlExt && subgroupSizeControlExtVersion >= 2)) {
    if (fpGetFeatures2 || fpGetFeatures2KHR) {
      vk::PhysicalDeviceFeatures2 features2{};
      vk::PhysicalDeviceSubgroupSizeControlFeatures sizeControlFeatures{};
      features2.pNext = &sizeControlFeatures;

      if (fpGetFeatures2) {
        fpGetFeatures2(
            static_cast<VkPhysicalDevice>(physicalDevice),
            reinterpret_cast<VkPhysicalDeviceFeatures2 *>(&features2));
      } else {
        fpGetFeatures2KHR(
            static_cast<VkPhysicalDevice>(physicalDevice),
            reinterpret_cast<VkPhysicalDeviceFeatures2 *>(&features2));
      }

      subgroupSizeControlFeature =
          static_cast<bool>(sizeControlFeatures.subgroupSizeControl);
    }
  } else if (hasSubgroupSizeControlExt) {
    // EXT version 1: the spec says to assume both features are supported.
    subgroupSizeControlFeature = true;
  }

  if (canQuerySizeControlProps) {
    out.controlProperties.maxComputeWorkgroupSubgroups =
        sizeControlProps.maxComputeWorkgroupSubgroups;

    out.controlProperties.supportedSubgroupSizes.clear();
    for (uint32_t sz = sizeControlProps.minSubgroupSize;
         sz <= sizeControlProps.maxSubgroupSize; sz *= 2) {
      out.controlProperties.supportedSubgroupSizes.push_back(sz);
    }

    out.controlProperties.supported =
        subgroupSizeControlFeature &&
        static_cast<bool>(sizeControlProps.requiredSubgroupSizeStages & vk::ShaderStageFlagBits::eCompute);
  } else {
    out.controlProperties.supportedSubgroupSizes = {out.subgroupSize};
  }

  return out;
}

} // namespace denox
