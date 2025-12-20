#pragma once

#include <stdexcept>
namespace denox::diag {
  
[[noreturn]] inline void invalid_state() {
  throw std::runtime_error("Invalid state");
}

}
