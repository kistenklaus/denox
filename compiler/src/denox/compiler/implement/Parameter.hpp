#pragma once

#include "denox/compiler/implement/TensorId.hpp"
#include "denox/memory/container/vector.hpp"
#include <cstddef>
#include <fmt/format.h>
#include <functional>
#include <memory>

namespace denox::compiler {

struct Parameter {
  TensorId tensorId;
  std::function<std::vector<std::byte>()> lazyValue;
};

} // namespace denox::compiler

template <> struct fmt::formatter<denox::compiler::Parameter> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::compiler::Parameter &p, FormatContext &ctx) const {
    return fmt::format_to(ctx.out(), "{{tensorId={}}}", p.tensorId.index);
  }
};
