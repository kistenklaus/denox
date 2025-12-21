#pragma once

#include "denox/common/ActivationFunction.hpp"
#include <fmt/core.h>

namespace denox::compiler {

struct ComputeOpActivation {
  ActivationFunction func;
};

} // namespace denox::compiler

template <> struct fmt::formatter<denox::compiler::ComputeOpActivation> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::compiler::ComputeOpActivation &op,
              FormatContext &ctx) const {
    return fmt::format_to(ctx.out(), "{{func={}}}", op.func);
  }
};
