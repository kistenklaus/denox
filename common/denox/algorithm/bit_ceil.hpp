#pragma once

#include <bit>

namespace denox::algorithm {

template <typename T> static constexpr inline T bit_ceil(T x) {
  return std::bit_ceil(x);
}

} // namespace denox::compiler
