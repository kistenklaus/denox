#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>

namespace denox::memory {
struct uint128 {
  uint64_t low = 0, high = 0;
  constexpr uint128() noexcept = default;
  constexpr uint128(uint64_t l, uint64_t h = 0) noexcept : low(l), high(h) {}
  constexpr explicit operator bool() const noexcept {
    return low != 0 || high != 0;
  }
  constexpr uint128 &operator+=(uint128 x) noexcept {
    const uint64_t oldLow = low;
    low += x.low;
    high += x.high + (low < oldLow ? 1 : 0);
    return *this;
  }
  constexpr uint128 &operator-=(uint128 x) noexcept {
    const uint64_t oldLow = low;
    low -= x.low;
    high -= x.high + (oldLow < x.low ? 1 : 0);
    return *this;
  }
  constexpr uint128 &operator&=(uint128 x) noexcept {
    low &= x.low;
    high &= x.high;
    return *this;
  }
  constexpr uint128 &operator|=(uint128 x) noexcept {
    low |= x.low;
    high |= x.high;
    return *this;
  }
  constexpr uint128 &operator^=(uint128 x) noexcept {
    low ^= x.low;
    high ^= x.high;
    return *this;
  }
  constexpr uint128 &operator<<=(std::size_t n) noexcept {
    if (n >= 128)
      low = high = 0;
    else if (n >= 64)
      high = low << (n - 64), low = 0;
    else if (n)
      high = (high << n) | (low >> (64 - n)), low <<= n;
    return *this;
  }
  constexpr uint128 &operator>>=(std::size_t n) noexcept {
    if (n >= 128)
      low = high = 0;
    else if (n >= 64)
      low = high >> (n - 64), high = 0;
    else if (n)
      low = (low >> n) | (high << (64 - n)), high >>= n;
    return *this;
  }
};
constexpr uint128 operator&(uint128 a, uint128 b) noexcept { return a &= b; }
constexpr uint128 operator|(uint128 a, uint128 b) noexcept { return a |= b; }
constexpr uint128 operator^(uint128 a, uint128 b) noexcept { return a ^= b; }
constexpr uint128 operator+(uint128 a, uint128 b) noexcept { return a += b; }
constexpr uint128 operator-(uint128 a, uint128 b) noexcept { return a -= b; }
constexpr uint128 operator~(uint128 a) noexcept { return {~a.low, ~a.high}; }
constexpr uint128 operator<<(uint128 a, std::size_t n) noexcept {
  return a <<= n;
}
constexpr uint128 operator>>(uint128 a, std::size_t n) noexcept {
  return a >>= n;
}
constexpr bool operator==(uint128 a, uint128 b) noexcept {
  return a.low == b.low && a.high == b.high;
}
constexpr bool operator!=(uint128 a, uint128 b) noexcept { return !(a == b); }
constexpr bool operator<(uint128 a, uint128 b) noexcept {
  return a.high < b.high || (a.high == b.high && a.low < b.low);
}
} // namespace denox::memory

namespace std {
template <> struct hash<denox::memory::uint128> {
  size_t operator()(const denox::memory::uint128 &x) const noexcept {
    uint64_t h = x.low;
    h ^= x.high + UINT64_C(0x9e3779b97f4a7c15) + (h << 6) + (h >> 2);
    return std::hash<uint64_t>{}(h);
  }
};
} // namespace std
