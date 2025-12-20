#pragma once

#include "denox/algorithm/hash_combine.hpp"
#include <cstdint>
#include <cstring>
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
