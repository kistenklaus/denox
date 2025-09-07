#pragma once

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
};

} // namespace denox::compiler
