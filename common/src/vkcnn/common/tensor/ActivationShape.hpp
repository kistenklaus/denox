#pragma once
#include <compare>

namespace vkcnn {

struct ActivationShape {
  unsigned int w;
  unsigned int h;
  unsigned int c;

  constexpr auto operator<=>(const ActivationShape &) const = default;
};

} // namespace vkcnn
