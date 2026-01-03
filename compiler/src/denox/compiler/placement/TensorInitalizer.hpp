#pragma once

#include <cstdint>
#include <vector>
namespace denox::compiler {

struct TensorInitializer {
  uint32_t tensor;
  std::vector<std::byte> data;
};

} // namespace denox::compiler
