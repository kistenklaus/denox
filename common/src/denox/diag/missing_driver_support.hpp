#pragma once

#include <stdexcept>

namespace denox::diag {
  
[[noreturn]] inline void missing_driver_support() {
  throw std::runtime_error("missing driver support");
}

}
