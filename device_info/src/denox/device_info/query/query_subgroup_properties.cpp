#include "denox/device_info/query/query_subgroup_properties.hpp"
#include "denox/device_info/SubgroupProperties.hpp"

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
  return out;
}

} // namespace denox::compiler::device_info::query
