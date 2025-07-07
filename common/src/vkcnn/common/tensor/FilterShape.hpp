#pragma once

#include <compare>

namespace vkcnn {

struct FilterShape {
  unsigned int s;
  unsigned int r;
  unsigned int c;
  unsigned int k;
  constexpr auto operator<=>(const FilterShape &) const = default;
};

} // namespace vkcnn
