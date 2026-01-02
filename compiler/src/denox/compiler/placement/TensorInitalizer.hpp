#pragma once

#include <cstdint>
#include <vector>
namespace denox::compiler {

struct TensorInitializers {
  uint32_t tensor;
  std::vector<std::byte> data;
};

} // namespace denox::compiler
