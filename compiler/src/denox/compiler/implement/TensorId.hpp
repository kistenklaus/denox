#pragma once

#include <cstdint>
#include <limits>
namespace denox::compiler {

struct TensorId {
  static constexpr std::uint64_t nullindex = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t index = nullindex;
};

}
