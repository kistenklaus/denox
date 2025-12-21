#pragma once

#include "denox/common/Lifetime.hpp"
#include "denox/memory/container/vector.hpp"
#include <fmt/core.h>

namespace denox::compiler {

struct Lifetimes {
  memory::vector<Lifetime> valueLifetimes;
};

} // namespace denox::compiler

template <> struct fmt::formatter<denox::Lifetime> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::Lifetime &lt, FormatContext &ctx) const {
    return fmt::format_to(ctx.out(), "[{}, {})", lt.start, lt.end);
  }
};
