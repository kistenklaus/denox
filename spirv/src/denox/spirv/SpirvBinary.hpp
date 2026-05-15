#pragma once

#include "denox/common/SHA256.hpp"
#include "denox/memory/container/vector.hpp"
#include <cstdint>

namespace denox {

struct SpirvBinary {
  memory::vector<std::uint32_t> spv;
  SHA256 source_hash;
};

} // namespace denox::compiler
