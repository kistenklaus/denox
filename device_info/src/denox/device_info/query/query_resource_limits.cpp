#include "denox/device_info/query/query_resource_limits.hpp"
#include "denox/device_info/ResourceLimits.hpp"
#include <cstring>

namespace denox {

ResourceLimits query_resource_limits([[maybe_unused]] vk::Instance instance,
                                     vk::PhysicalDevice physicalDevice) {
  ResourceLimits limits;

  vk::PhysicalDeviceLimits deviceLimits = physicalDevice.getProperties().limits;

  std::memcpy(limits.maxComputeWorkGroupCount.data(),
              deviceLimits.maxComputeWorkGroupCount.data(),
              deviceLimits.maxComputeWorkGroupCount.size() *
                  sizeof(std::uint32_t));
  std::memcpy(limits.maxComputeWorkGroupSize.data(),
              deviceLimits.maxComputeWorkGroupSize.data(),
              deviceLimits.maxComputeWorkGroupSize.size() *
                  sizeof(std::uint32_t));
  limits.maxComputeWorkGroupInvocations =
      deviceLimits.maxComputeWorkGroupInvocations;
  limits.maxComputeSharedMemory = deviceLimits.maxComputeSharedMemorySize;
  limits.maxPerStageResources = deviceLimits.maxPerStageResources;
  limits.maxPerStageUniformBuffers =
      deviceLimits.maxPerStageDescriptorUniformBuffers;
  limits.maxPerStageStorageBuffers =
      deviceLimits.maxPerStageDescriptorStorageBuffers;
  limits.maxPerStageSampledImages =
      deviceLimits.maxPerStageDescriptorSampledImages;
  limits.maxPerStageStorageImages =
      deviceLimits.maxPerStageDescriptorStorageImages;
  limits.maxPerStageSamplers = deviceLimits.maxPerStageDescriptorSamplers;
  limits.maxPushConstantSize = deviceLimits.maxPushConstantsSize;

  return limits;
}

} // namespace denox::compiler::device_info::query
