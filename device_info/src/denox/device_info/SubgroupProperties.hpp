#pragma once

#include "denox/memory/container/small_vector.hpp"
#include <cstdint>

namespace denox {

struct SubgroupControlProperties {
  bool supported;
  memory::small_vector<uint32_t, 2> supportedSubgroupSizes;
  uint32_t maxComputeWorkgroupSubgroups;
};

struct SubgroupProperties {
  std::uint32_t subgroupSize;
  bool supportsBasicOps;
  bool supportsVoteOps;
  bool supportsArithmeticOps;
  bool supportsBallotOps;
  bool supportsShuffleOps;
  bool supportsShuffleRelativeOps;

  SubgroupControlProperties controlProperties;
};

} // namespace denox
