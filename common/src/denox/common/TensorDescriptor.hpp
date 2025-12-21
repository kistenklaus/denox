#pragma once

#include "denox/common/TensorDataType.hpp"
#include "denox/common/TensorFormat.hpp"
#include "denox/common/TensorStorage.hpp"
#include "denox/symbolic/Sym.hpp"
#include <fmt/core.h>

namespace denox {

struct TensorDescriptor {
  Sym width;
  Sym height;
  Sym channels;
  TensorStorage storage;
  TensorFormat format;
  TensorDataType type;
};

} // namespace denox

template <> struct fmt::formatter<denox::TensorDescriptor> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::TensorDescriptor &t, FormatContext &ctx) const {
    return fmt::format_to(ctx.out(),
                          "{{w={}, h={}, c={}, storage={}, format={}, "
                          "dtype={}}}",
                          t.width, t.height, t.channels, t.storage, t.format,
                          t.type);
  }
};
