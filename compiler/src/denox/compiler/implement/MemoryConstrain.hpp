#pragma once

#include "denox/compiler/implement/TensorId.hpp"
#include <fmt/format.h>

namespace denox::compiler {

struct MemoryImplicitConcatConstrain {
  // NOTE: Place tensorA and tensorB behind each other in memory.
  TensorId src0;
  TensorId src1;
  TensorId dst;
};

} // namespace denox::compiler

template <>
struct fmt::formatter<denox::compiler::MemoryImplicitConcatConstrain> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::compiler::MemoryImplicitConcatConstrain &c,
              FormatContext &ctx) const {
    return fmt::format_to(ctx.out(), "{{src0={}, src1={}, dst={}}}",
                          c.src0.index, c.src1.index, c.dst.index);
  }
};
