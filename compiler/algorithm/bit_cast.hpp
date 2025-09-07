#pragma once

#include <bit>

namespace denox::algorithm {

template <class To, class From>
constexpr To bit_cast(const From &from) noexcept {
  return std::bit_cast<To>(from);
}

} // namespace denox::algorithms
