#pragma once

#include "denox/memory/dtype/dtype.hpp"
#include "denox/memory/tensor/FilterLayout.hpp"
#include "denox/memory/tensor/FilterShape.hpp"
#include <fmt/core.h>

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

} // namespace denox::memory

template <> struct fmt::formatter<denox::memory::FilterDescriptor> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::memory::FilterDescriptor &fd,
              FormatContext &ctx) const {
    return fmt::format_to(ctx.out(), "{{shape={}, layout={}, type={}}}",
                          fd.shape, fd.layout, fd.type);
  }
};
