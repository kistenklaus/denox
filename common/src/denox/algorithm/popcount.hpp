#pragma once

#include "denox/memory/container/uint128.hpp"

#include <bit>
#include <concepts>

namespace denox::algorithm {

template <std::unsigned_integral T>
constexpr int popcount(T value) noexcept {
  return std::popcount(value);
}

constexpr int popcount(memory::uint128 value) noexcept {
  return std::popcount(value.low) + std::popcount(value.high);
}

} // namespace denox::algorithm
