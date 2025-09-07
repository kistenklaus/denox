#pragma once

#include <stdexcept>
namespace denox::compiler::diag {

[[noreturn]] static inline void unreachable() {
  throw std::logic_error("unrechable");
}

} // namespace denox::compiler
