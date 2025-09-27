#pragma once

#include <cstdint>
namespace denox::compiler {

struct ResourceLimits {
  std::uint32_t maxComputeWorkGroupCount[3];
  std::uint32_t maxComputeWorkGroupSize[3];
  std::uint32_t maxComputeWorkGroupInvocations;
  std::uint32_t maxComputeSharedMemory;

  std::uint32_t maxPerStageResources;
  std::uint32_t maxPerStageUniformBuffers;
  std::uint32_t maxPerStageStorageBuffers;
  std::uint32_t maxPerStageSampledImages;
  std::uint32_t maxPerStageStorageImages;
  std::uint32_t maxPerStageSamplers;

  std::uint32_t maxPushConstantSize;
};

}
