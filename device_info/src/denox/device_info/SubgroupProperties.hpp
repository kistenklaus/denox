#pragma once

#include <cstdint>

namespace denox {

struct SubgroupProperties {
  std::uint32_t subgroupSize;
  bool supportsBasicOps;
  bool supportsVoteOps;
  bool supportsArithmeticOps;
  bool supportsBallotOps;
  bool supportsShuffleOps;
  bool supportsShuffleRelativeOps;
};

}
