#pragma once

#include "denox/memory/container/vector.hpp"
#include <cstdint>
#include <fmt/core.h>

namespace denox::compiler {

struct Lifetime {
  std::uint64_t start;
  std::uint64_t end;
};

struct Lifetimes {
  memory::vector<Lifetime> valueLifetimes;
};

} // namespace denox::compiler

template <> struct fmt::formatter<denox::compiler::Lifetime> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::compiler::Lifetime &lt, FormatContext &ctx) const {
    return fmt::format_to(ctx.out(), "[{}, {})", lt.start, lt.end);
  }
};
