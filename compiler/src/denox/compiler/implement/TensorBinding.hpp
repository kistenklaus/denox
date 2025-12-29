#pragma once

#include "denox/common/Access.hpp"
#include "denox/compiler/implement/TensorId.hpp"
#include <fmt/format.h>

namespace denox::compiler {

struct TensorBinding {
  std::uint32_t set;
  std::uint32_t binding;
  Access accessFlag;
  TensorId tensorId;
};

} // namespace denox::compiler

template <> struct fmt::formatter<denox::compiler::TensorBinding> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::compiler::TensorBinding &tb,
              FormatContext &ctx) const {
    return fmt::format_to(ctx.out(),
                          "{{set={}, binding={}, accessFlag={}, tensorId={}}}",
                          tb.set, tb.binding, tb.accessFlag, tb.tensorId.index);
  }
};
