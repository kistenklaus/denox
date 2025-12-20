#pragma once

#include <cassert>
#include <stdexcept>

namespace denox::diag {
  
[[noreturn]] inline void not_implemented() {
  assert(false);
  throw std::runtime_error("Not implemented");
}

}
