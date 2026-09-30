#pragma once

#include "denox/memory/container/uint128.hpp"
#include <bit>
#include <concepts>
namespace denox::algorithm {

template <std::unsigned_integral Int>
constexpr int countr_zero(Int v) noexcept {
  return std::countr_zero(v);
}

constexpr int countr_zero(memory::uint128 v) noexcept {
  if (v.low != 0)
    return std::countr_zero(v.low);
  if (v.high != 0)
    return 64 + std::countr_zero(v.high);
  return 128;
}

} // namespace denox::algorithm
