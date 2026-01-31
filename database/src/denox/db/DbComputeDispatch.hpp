#pragma once

#include "denox/db/DbTensorBinding.hpp"
#include "denox/db/DbTiming.hpp"
#include "denox/memory/container/optional.hpp"
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

  // extra info for data science.
  memory::optional<std::string> operation;
  memory::optional<std::string> shader_name;
  memory::optional<std::string> config;
  memory::optional<uint64_t> memory_reads;
  memory::optional<uint64_t> memory_writes;
  memory::optional<uint64_t> flops;
  bool coopmat;
};
} // namespace denox
