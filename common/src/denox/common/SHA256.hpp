#pragma once

#include "denox/algorithm/hash_combine.hpp"
#include <cstdint>
#include <cstring>
#include <fmt/format.h>
#include <functional>
#include <span>

namespace denox {

struct SHA256 {
  uint32_t h[8];
  friend bool operator==(const SHA256 &lhs, const SHA256 &rhs) {
    return std::memcmp(lhs.h, rhs.h, 8 * sizeof(uint32_t)) == 0;
  }

  friend bool operator!=(const SHA256 &lhs, const SHA256 &rhs) {
    return std::memcmp(lhs.h, rhs.h, 8 * sizeof(uint32_t)) != 0;
  }
};

struct SHA256Builder {
  SHA256Builder();

  void update(std::span<const uint8_t> data);
  void update_uint32(uint32_t v);
  void update_uint64(uint64_t v);

  SHA256 finalize();

private:
  void process_block(const uint8_t block[64]);

  uint32_t h[8];
  uint8_t buffer[64];
  uint64_t bit_len;
  uint32_t buffer_len;
};

} // namespace denox

template <> struct std::hash<denox::SHA256> {
  size_t operator()(const denox::SHA256 &v) const noexcept {
    size_t hash = 0;
    for (uint32_t x : v.h) {
      hash = denox::algorithm::hash_combine(hash, x);
    }
    return hash;
  }
};

template <> struct fmt::formatter<denox::SHA256> {
  // number of hex chars to print (must be even)
  size_t hex_chars = 8;

  constexpr auto parse(format_parse_context &ctx) {
    auto it = ctx.begin();
    auto end = ctx.end();

    // Optional custom length: {:12} → first 12 hex chars
    if (it != end && *it >= '0' && *it <= '9') {
      size_t v = 0;
      while (it != end && *it >= '0' && *it <= '9') {
        v = v * 10 + (static_cast<size_t>(*it) - static_cast<size_t>('0'));
        ++it;
      }
      if (v > 0) {
        hex_chars = v & ~size_t(1); // force even
      }
    }

    if (it != end && *it != '}')
      throw format_error("invalid format");

    return it;
  }

  template <typename FormatContext>
  auto format(const denox::SHA256 &v, FormatContext &ctx) const {
    static constexpr char hex[] = "0123456789abcdef";

    auto out = ctx.out();
    const uint8_t *bytes = reinterpret_cast<const uint8_t *>(v.h);

    const size_t byte_count = std::min(hex_chars / 2, size_t(32));
    for (size_t i = 0; i < byte_count; ++i) {
      uint8_t b = bytes[i];
      *out++ = hex[b >> 4];
      *out++ = hex[b & 0xF];
    }

    return out;
  }
};
