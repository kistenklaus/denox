#pragma once

#include <fmt/format.h>
#include <stdexcept>
namespace denox::diag {
  
[[noreturn]] inline void invalid_state(const std::string& msg = {}) {
  if (msg.empty()) {
    throw std::runtime_error("Invalid state");
  } else {
    throw std::runtime_error(fmt::format("Invalid state: {}", msg));
  }
}

}
