#pragma once
#include <fmt/core.h>

namespace denox {

enum class ActivationFunction {
  ReLU,
  LeakyReLU,
  SiLU,
};

}

template <> struct fmt::formatter<denox::ActivationFunction> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(denox::ActivationFunction f, FormatContext &ctx) const {
    using enum denox::ActivationFunction;

    const char *name = nullptr;
    switch (f) {
    case ReLU:
      name = "ReLU";
      break;
    case LeakyReLU:
      name = "LeakyReLU";
      break;
    case SiLU:
      name = "SiLU";
      break;
    }

    return fmt::format_to(ctx.out(), "{}", name);
  }
};
