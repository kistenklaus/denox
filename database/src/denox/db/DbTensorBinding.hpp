#pragma once

#include "denox/common/Access.hpp"
#include <cstdint>
namespace denox {

struct DbTensorBinding {
  uint32_t set;
  uint32_t binding;
  Access access;
  uint64_t byteSize;
  uint16_t alignment;
};

}
