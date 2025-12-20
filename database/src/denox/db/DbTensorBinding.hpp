#pragma once

#include "denox/common/Access.hpp"
#include <cstdint>
namespace denox {

struct DbTensorBinding {
  uint32_t set;
  uint32_t binding;
  Access access;
  uint32_t byteSize;
  uint32_t alignment;
};

}
