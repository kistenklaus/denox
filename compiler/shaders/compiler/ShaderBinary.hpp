#pragma once

#include "memory/container/vector.hpp"
#include <cstdint>
namespace denox::compiler {

struct ShaderBinary {
  memory::vector<std::uint32_t> spv;
};

} // namespace denox::compiler
