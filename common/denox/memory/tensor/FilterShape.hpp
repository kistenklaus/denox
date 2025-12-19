#pragma once

#include <cstddef>
namespace denox::memory {

struct FilterShape {
  unsigned int s;
  unsigned int r;
  unsigned int c;
  unsigned int k;

  friend bool operator==(const FilterShape &lhs, const FilterShape &rhs) {
    return lhs.s == rhs.s && lhs.r == rhs.r && lhs.c == rhs.c && lhs.k == rhs.k;
  }
  friend bool operator!=(const FilterShape &lhs, const FilterShape &rhs) {
    return !(lhs == rhs);
  }

  std::size_t elemCount() const {
    return static_cast<std::size_t>(s) * static_cast<std::size_t>(r) *
           static_cast<std::size_t>(c) * static_cast<std::size_t>(k);
  }
};

} // namespace denox::memory
