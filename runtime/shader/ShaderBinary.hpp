#pragma once

#include <cstdint>
#include <vector>

namespace denox::glsl {

struct ShaderBinary {
  std::vector<std::uint32_t> spv;
};

} // namespace denox::compiler
