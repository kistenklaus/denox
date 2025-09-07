#pragma once

#include "memory/dtype/dtype.hpp"
#include "memory/tensor/BiasLayout.hpp"

namespace denox::memory {

struct BiasDescriptor {
  unsigned int shape;
  BiasLayout layout;
  Dtype type;

  std::size_t byteSize() const { return layout.size(shape) * type.size(); }

  friend bool operator==(const BiasDescriptor &lhs, const BiasDescriptor &rhs) {
    return lhs.shape == rhs.shape && lhs.layout == rhs.layout &&
           lhs.type == rhs.type;
  }

  friend bool operator!=(const BiasDescriptor &lhs, const BiasDescriptor &rhs) {
    return !(lhs == rhs);
  }
};

} // namespace denox::compiler
