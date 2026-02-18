#include "denox/device_info/query/query_subgroup_properties.hpp"
#include "denox/device_info/SubgroupProperties.hpp"
#include <vulkan/vulkan_core.h>

namespace denox {

SubgroupProperties
query_subgroup_properties([[maybe_unused]] vk::Instance instance,
                          vk::PhysicalDevice physicalDevice) {

  SubgroupProperties out;
  { // query subgroup properties.
    vk::PhysicalDeviceSubgroupProperties subgroupProperties;
    vk::PhysicalDeviceProperties2 prop2;
    prop2.pNext = &subgroupProperties;
    physicalDevice.getProperties2(&prop2);

    out.subgroupSize = subgroupProperties.subgroupSize;
    if (!(subgroupProperties.supportedStages &
          vk::ShaderStageFlagBits::eCompute)) {
      out.supportsBasicOps = false;
      out.supportsVoteOps = false;
      out.supportsArithmeticOps = false;
      out.supportsBallotOps = false;
      out.supportsShuffleOps = false;
      out.supportsShuffleRelativeOps = false;
    } else {
      out.supportsBasicOps =
          static_cast<bool>(subgroupProperties.supportedOperations &
                            vk::SubgroupFeatureFlagBits::eBasic);
      out.supportsVoteOps =
          static_cast<bool>(subgroupProperties.supportedOperations &
                            vk::SubgroupFeatureFlagBits::eVote);
      out.supportsArithmeticOps =
          static_cast<bool>(subgroupProperties.supportedOperations &
                            vk::SubgroupFeatureFlagBits::eArithmetic);
      out.supportsBallotOps =
          static_cast<bool>(subgroupProperties.supportedOperations &
                            vk::SubgroupFeatureFlagBits::eBallot);
      out.supportsShuffleOps =
          static_cast<bool>(subgroupProperties.supportedOperations &
                            vk::SubgroupFeatureFlagBits::eShuffle);
      out.supportsShuffleRelativeOps =
          static_cast<bool>(subgroupProperties.supportedOperations &
                            vk::SubgroupFeatureFlagBits::eShuffleRelative);
    }
  }

#ifdef VK_VERSION_1_3
  {
    VkPhysicalDeviceSubgroupSizeControlProperties props{};
    props.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
    VkPhysicalDeviceProperties2 props2;
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &props;
    vkGetPhysicalDeviceProperties2(physicalDevice, &props2);

    out.controlProperties.maxComputeWorkgroupSubgroups =
        props.maxComputeWorkgroupSubgroups;
    for (uint32_t sz = props.minSubgroupSize; sz <= props.maxSubgroupSize;
         sz *= 2) {
      out.controlProperties.supportedSubgroupSizes.push_back(sz);
    }

    out.controlProperties.supported =
        props.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT;
  }
#else
  {
    out.controlProperties.maxComputeWorkgroupSubgroups = 100000;
    out.controlProperties.supportedSubgroupSizes = {out.subgroupSize};
    out.controlProperties.supported = false;
  }
#endif

  return out;
}

} // namespace denox
