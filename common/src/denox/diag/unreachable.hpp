#pragma once

#include "assume.hpp"
#include <fmt/format.h>
#include <stdexcept>

namespace denox::diag {

[[noreturn]] static inline void unreachable(const std::string& msg = {}) {
#ifndef NDEBUG
  if (msg.empty()) {
    throw std::logic_error("unrechable");
  } else {
    throw std::logic_error(fmt::format("unrechable: {}", msg));
  }
#else
#if defined(_MSC_VER) && !defined(__clang__) // MSVC
  __assume(false);
#else                                        // GCC, Clang
  __builtin_unreachable();
#endif
#endif
}

} // namespace denox::compiler::diag
