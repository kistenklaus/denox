#pragma once

#include <array>
#include <cstdint>

namespace denox {

struct ResourceLimits {
  std::array<std::uint32_t, 3> maxComputeWorkGroupCount;
  std::array<std::uint32_t, 3> maxComputeWorkGroupSize;
  std::uint32_t maxComputeWorkGroupInvocations;
  std::uint32_t maxComputeSharedMemory;
  std::uint32_t maxPushConstantSize;
};

}
