#pragma once

#include "model/FilterMode.hpp"
namespace denox::compiler { 

struct ComputeOpUpsample {
  unsigned int scalingFactor;
  FilterMode mode;
};

}
