#pragma once

#include <fmt/format.h>
#include <stdexcept>
namespace denox::diag {

[[noreturn]] inline void invalid_argument(std::string_view msg = {}) {
  if (msg.empty()) {
    throw std::runtime_error("Invalid Argument");
  } else {
    throw std::runtime_error(fmt::format("Invalid Argument: {}", msg));
  }
}

} // namespace denox::diag
