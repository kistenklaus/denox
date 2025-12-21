#pragma once

#include <fmt/core.h>

namespace denox::compiler {

struct ComputeOpConcat {};

} // namespace vkcnn

template <>
struct fmt::formatter<denox::compiler::ComputeOpConcat> {
  constexpr auto parse(fmt::format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const denox::compiler::ComputeOpConcat&,
              FormatContext& ctx) const {
    return fmt::format_to(ctx.out(), "concat");
  }
};
