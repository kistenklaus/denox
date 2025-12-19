#pragma once

#include "assume.hpp"
#include <stdexcept>

namespace denox::compiler::diag {

[[noreturn]] static inline void unreachable() {
#ifndef NDEBUG
  throw std::logic_error("unrechable");
#else
#if defined(_MSC_VER) && !defined(__clang__) // MSVC
  __assume(false);
#else                                        // GCC, Clang
  __builtin_unreachable();
#endif
#endif
}

} // namespace denox::compiler::diag
