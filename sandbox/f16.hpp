#pragma once

#include <bit>
#include <cstdint>
struct f16 {
  f16() : m_bits() {}
  explicit f16(float v) : m_bits(from_float(v)) {}
  explicit f16(double v) : m_bits(from_float(static_cast<float>(v))) {}

  static f16 Raw(std::uint16_t bits) {
    f16 x;
    x.m_bits = bits;
    return x;
  }

  explicit operator float() const { return to_float(m_bits); }
  explicit operator double() const {
    return static_cast<double>(to_float(m_bits));
  }

private:
  static std::uint16_t from_float(float f) {
    std::uint32_t x = std::bit_cast<uint32_t>(f);
    std::uint32_t sign = (x >> 16) & 0x8000;
    std::uint32_t mantissa = x & 0x007FFFFF;
    std::int32_t exponent =
        ((static_cast<std::int32_t>(x) >> 23) & 0xFF) - 127 + 15;

    if (exponent <= 0) {
      return static_cast<uint16_t>(sign);
    } else if (exponent >= 31) {
      return static_cast<uint16_t>(sign | 0x7C00); // Inf
    }
    return static_cast<uint16_t>(
        sign | (static_cast<std::uint32_t>(exponent) << 10) | (mantissa >> 13));
  }

  static float to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x03FF;

    if (exponent == 0) {
      // Subnormal or zero
      if (mantissa == 0) {
        uint32_t bits = sign;
        return std::bit_cast<float>(bits);
      }
      // Normalize subnormal
      exponent = 1;
      while ((mantissa & 0x0400) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x3FF;
    } else if (exponent == 31) {
      // Inf or NaN
      uint32_t bits = sign | 0x7F800000 | (mantissa << 13);
      return std::bit_cast<float>(bits);
    }

    exponent = exponent - 15 + 127;
    uint32_t bits = sign | (exponent << 23) | (mantissa << 13);
    return std::bit_cast<float>(bits);
  }

  std::uint16_t m_bits;
};
static_assert(sizeof(f16) == 2);
