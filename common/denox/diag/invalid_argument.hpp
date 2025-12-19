#pragma once

#include <stdexcept>
namespace denox::compiler::diag {

[[noreturn]] inline void invalid_argument() {
  throw std::runtime_error("invalid_argument");
}

} // namespace denox::compiler
