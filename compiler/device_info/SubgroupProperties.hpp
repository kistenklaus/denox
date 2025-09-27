#pragma once

#include <cstdint>
namespace denox::compiler {

struct SubgroupProperties {
  std::uint32_t subgroupSize;
  bool supportsBasicOps;
  bool supportsVoteOps;
  bool supportsArithmeticOps;
  bool supportsBallotOps;
  bool supportsShuffleOps;
  bool supportsShuffleRelativeOps;

  bool subgroupControl;
  bool subgroupControlSupportsComputeFullSubgroups;
};

}
