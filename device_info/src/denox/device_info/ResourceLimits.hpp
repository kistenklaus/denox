#pragma once

#include <array>
#include <cstdint>

namespace denox {

struct ResourceLimits {
  std::array<std::uint32_t, 3> maxComputeWorkGroupCount;
  std::array<std::uint32_t, 3> maxComputeWorkGroupSize;
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
