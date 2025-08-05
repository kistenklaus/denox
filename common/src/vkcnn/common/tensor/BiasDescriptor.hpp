#pragma once

#include "vkcnn/common/tensor/BiasLayout.hpp"
#include "vkcnn/common/tensor/FloatType.hpp"
#include <compare>

namespace vkcnn {

struct BiasDescriptor {
  unsigned int shape;
  BiasLayout layout;
  FloatType type;

  std::size_t byteSize() const { return layout.size(shape) * type.size(); }

  constexpr auto operator<=>(const BiasDescriptor &) const = default;
};

} // namespace vkcnn
