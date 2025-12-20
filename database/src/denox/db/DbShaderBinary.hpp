#pragma once

#include "denox/common/SHA256.hpp"
#include "denox/spirv/SpirvBinary.hpp"
namespace denox {

struct DbShaderBinary {
  SHA256 hash;
  SpirvBinary spvBinary;
};

} // namespace denox
