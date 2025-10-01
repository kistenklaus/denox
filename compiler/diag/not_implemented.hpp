#pragma once

#include <stdexcept>
namespace denox::compiler::diag {
  
[[noreturn]] inline void not_implemented() {
  throw std::runtime_error("Not implemented");
}

}
