#pragma once

#include <cstddef>
namespace denox::algorithm {

inline std::size_t hash_combine(std::size_t lhs, std::size_t rhs) {
  std::size_t hash = lhs;
  hash ^= rhs + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  return hash;
}

} // namespace denox::algorithm
