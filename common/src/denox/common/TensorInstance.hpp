#pragma once

#include "denox/common/ComputeOp.hpp"
#include "denox/common/Lifetime.hpp"
#include "denox/common/TensorDataType.hpp"
#include "denox/common/TensorFormat.hpp"
#include "denox/common/TensorStorage.hpp"
#include "denox/memory/hypergraph/LinkedGraph.hpp"
#include "denox/symbolic/Sym.hpp"
#include <fmt/core.h>

namespace denox {

struct TensorInstance {
  Sym width;
  Sym height;
  Sym channels;
  TensorStorage storage;
  TensorFormat format;
  TensorDataType type;
  memory::LinkedGraph<TensorInstance, ComputeOp>::NodeHandle originalNode;
  Lifetime lifetime;
};

} // namespace denox

template <> struct fmt::formatter<denox::TensorInstance> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::TensorInstance &t, FormatContext &ctx) const {
    return fmt::format_to(ctx.out(),
                          "{{w={}, h={}, c={}, storage={}, format={}, "
                          "dtype={}, vid=N{}, lifetime={}}}",
                          t.width, t.height, t.channels, t.storage, t.format,
                          t.type, *t.originalNode->id(), t.lifetime);
  }
};
