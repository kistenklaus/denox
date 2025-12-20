#pragma once

#include "denox/common/FilterMode.hpp"

namespace denox::compiler { 

struct ComputeOpUpsample {
  unsigned int scalingFactor;
  FilterMode mode;
};

}
