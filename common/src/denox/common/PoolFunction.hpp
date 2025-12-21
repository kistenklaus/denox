#pragma once

#include <fmt/core.h>

namespace denox {

enum class PoolFunction {
  Max,
  Avg,
};

}

template <> struct fmt::formatter<denox::PoolFunction> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(denox::PoolFunction f, FormatContext &ctx) const {
    const char *name = nullptr;
    switch (f) {
    case denox::PoolFunction::Max:
      name = "max";
      break;
    case denox::PoolFunction::Avg:
      name = "avg";
      break;
    }
    return fmt::format_to(ctx.out(), "{}", name);
  }
};
