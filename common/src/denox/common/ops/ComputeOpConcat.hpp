#pragma once

#include <fmt/core.h>

namespace denox {

struct ComputeOpConcat {};

} // namespace vkcnn

template <>
struct fmt::formatter<denox::ComputeOpConcat> {
  constexpr auto parse(fmt::format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const denox::ComputeOpConcat&,
              FormatContext& ctx) const {
    return fmt::format_to(ctx.out(), "concat");
  }
};
