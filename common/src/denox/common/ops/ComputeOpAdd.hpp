#pragma once

#include <fmt/core.h>

namespace denox {

struct ComputeOpAdd {};

} // namespace denox

template <> struct fmt::formatter<denox::ComputeOpAdd> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::ComputeOpAdd &, FormatContext &ctx) const {
    return fmt::format_to(ctx.out(), "add");
  }
};
