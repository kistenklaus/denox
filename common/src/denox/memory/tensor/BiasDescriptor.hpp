#pragma once

#include "denox/memory/dtype/dtype.hpp"
#include "denox/memory/tensor/BiasLayout.hpp"
#include <fmt/core.h>

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

} // namespace denox::memory

template <> struct fmt::formatter<denox::memory::BiasDescriptor> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::memory::BiasDescriptor &bd,
              FormatContext &ctx) const {
    return fmt::format_to(ctx.out(), "{{shape={}, layout={}, type={}}}",
                          bd.shape, bd.layout, bd.type);
  }
};
