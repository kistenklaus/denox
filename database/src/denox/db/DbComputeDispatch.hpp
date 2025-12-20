#pragma once

#include "denox/db/DbTensorBinding.hpp"
#include "denox/db/DbTiming.hpp"
#include <cstdint>
#include <optional>
#include <vector>
namespace denox {

struct DbComputeDispatch {
  uint32_t binaryId;
  uint32_t workgroupCountX;
  uint32_t workgroupCountY;
  uint32_t workgroupCountZ;
  std::vector<uint8_t> pushConstant;
  uint64_t hash;
  std::vector<DbTensorBinding> bindings;
  std::optional<DbDispatchTiming> time;
};

}
