#pragma once

#include "vkcnn/common/FilterMode.hpp"
namespace vkcnn { 

struct ComputeOpUpsample {
  unsigned int scalingFactor;
  FilterMode mode;
};

}
