#pragma once

#include "denox/memory/container/vector.hpp"
#include <cstdint>

namespace denox {

struct SpirvBinary {
  memory::vector<std::uint32_t> spv;
};

} // namespace denox::compiler
