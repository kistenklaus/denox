#pragma once

#include "denox/common/FilterMode.hpp"
#include <fmt/core.h>

namespace denox::compiler { 

struct ComputeOpUpsample {
  unsigned int scalingFactor;
  FilterMode mode;
};

}


template <>
struct fmt::formatter<denox::compiler::ComputeOpUpsample> {
  constexpr auto parse(fmt::format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const denox::compiler::ComputeOpUpsample& op,
              FormatContext& ctx) const {
    return fmt::format_to(
        ctx.out(),
        "{{scale={}, mode={}}}",
        op.scalingFactor,
        op.mode);
  }
};
