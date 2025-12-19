#pragma once

#include <stdexcept>
namespace denox::compiler::diag {
  
[[noreturn]] inline void invalid_state() {
  throw std::runtime_error("Invalid state");
}

}
