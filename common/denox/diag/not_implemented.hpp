#pragma once

#include <cassert>
#include <stdexcept>
namespace denox::compiler::diag {
  
[[noreturn]] inline void not_implemented() {
  assert(false);
  throw std::runtime_error("Not implemented");
}

}
