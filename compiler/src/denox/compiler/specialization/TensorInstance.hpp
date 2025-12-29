#pragma once

#include "denox/common/Lifetime.hpp"
#include "denox/compiler/canonicalize/CanoModel.hpp"
#include <fmt/core.h>

namespace denox::compiler {

struct TensorInstance {
  Sym width;
  Sym height;
  Sym channels;
  TensorStorage storage;
  TensorFormat format;
  TensorDataType type;
  CanoModel::Graph::NodeHandle originalNode;
  Lifetime lifetime;

  std::uint64_t valueId() const { return *originalNode->id(); }
};

} // namespace denox::compiler

template <> struct fmt::formatter<denox::compiler::TensorInstance> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::compiler::TensorInstance &t,
              FormatContext &ctx) const {
    return fmt::format_to(ctx.out(),
                          "{{w={}, h={}, c={}, storage={}, format={}, "
                          "dtype={}, vid=N{}, lifetime={}}}",
                          t.width, t.height, t.channels, t.storage, t.format,
                          t.type, *t.originalNode->id(), t.lifetime);
  }
};
