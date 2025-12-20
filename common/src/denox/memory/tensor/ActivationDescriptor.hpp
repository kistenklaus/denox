#pragma once

#include "denox/memory/dtype/dtype.hpp"
#include "denox/memory/tensor/ActivationLayout.hpp"
#include "denox/memory/tensor/ActivationShape.hpp"

namespace denox::memory {

struct ActivationDescriptor {
  ActivationShape shape;
  ActivationLayout layout;
  Dtype type;

  std::size_t byteSize() const {
    return shape.c * shape.w * shape.h * type.size();
  }

  friend bool operator==(const ActivationDescriptor &lhs,
                         const ActivationDescriptor &rhs) {
    return lhs.shape == rhs.shape && lhs.layout == rhs.layout &&
           lhs.type == rhs.type;
  }

  friend bool operator!=(const ActivationDescriptor &lhs,
                         const ActivationDescriptor &rhs) {
    return !(lhs == rhs);
  }
};

} // namespace denox::compiler
