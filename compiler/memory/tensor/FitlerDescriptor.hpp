#pragma once

#include "memory/dtype/dtype.hpp"
#include "memory/tensor/FilterLayout.hpp"
#include "memory/tensor/FilterShape.hpp"

namespace denox::memory {

struct FilterDescriptor {
  FilterShape shape;
  FilterLayout layout;
  Dtype type;

  std::size_t byteSize() const {
    return shape.c * shape.k * shape.r * shape.s * type.size();
  }

  friend bool operator==(const FilterDescriptor &lhs,
                         const FilterDescriptor &rhs) {
    return lhs.shape == rhs.shape && lhs.layout == rhs.layout &&
           lhs.type == rhs.type;
  }

  friend bool operator!=(const FilterDescriptor &lhs,
                         const FilterDescriptor &rhs) {
    return !(lhs == rhs);
  }
};

} // namespace denox::compiler
