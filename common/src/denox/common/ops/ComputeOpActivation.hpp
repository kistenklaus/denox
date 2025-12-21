#pragma once

#include "denox/common/ActivationFunction.hpp"
#include <fmt/core.h>

namespace denox {

struct ComputeOpActivation {
  ActivationFunction func;
};

} // namespace denox::compiler

template <> struct fmt::formatter<denox::ComputeOpActivation> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::ComputeOpActivation &op,
              FormatContext &ctx) const {
    return fmt::format_to(ctx.out(), "{{func={}}}", op.func);
  }
};
