#pragma once

#include "vkcnn/common/tensor/FilterLayout.hpp"
#include "vkcnn/common/tensor/FilterShape.hpp"
#include "vkcnn/common/tensor/FloatType.hpp"
#include <compare>
namespace vkcnn {

struct FilterDescriptor {
  FilterShape shape;
  FilterLayout layout;
  FloatType type;

  std::size_t byteSize() const {
    return shape.c * shape.k * shape.r * shape.s * type.size();
  }

  constexpr auto operator<=>(const FilterDescriptor &) const = default;
};

} // namespace vkcnn
