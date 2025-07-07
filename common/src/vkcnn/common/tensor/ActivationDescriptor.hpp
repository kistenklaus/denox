#pragma once

#include "vkcnn/common/tensor/ActivationLayout.hpp"
#include "vkcnn/common/tensor/ActivationShape.hpp"
#include "vkcnn/common/tensor/FloatType.hpp"
#include <compare>

namespace vkcnn {

struct ActivationDescriptor {
  ActivationShape shape;
  ActivationLayout layout;
  FloatType type;

  std::size_t byteSize() const {
    return shape.c * shape.w * shape.h * type.size();
  }

  constexpr auto operator<=>(const ActivationDescriptor &) const = default;
};

} // namespace vkcnn
