#pragma once

#include "denox/common/TensorDataType.hpp"
#include "denox/common/TensorFormat.hpp"
#include "denox/common/TensorStorage.hpp"
#include "denox/memory/container/optional.hpp"
#include "denox/memory/container/string.hpp"
#include "denox/symbolic/Sym.hpp"
#include <fmt/format.h>

namespace denox::compiler {

struct TensorInfo {
  memory::optional<memory::string> name;
  memory::optional<Sym> width;
  memory::optional<Sym> height;
  memory::optional<Sym> channels;
  TensorStorage storage;
  TensorFormat format;
  TensorDataType type;
};

struct Tensor {
  Sym size;
  uint16_t alignment;
  TensorInfo info;
};

} // namespace denox::compiler

template <> struct fmt::formatter<denox::compiler::TensorInfo> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::compiler::TensorInfo &info,
              FormatContext &ctx) const {
    auto out = ctx.out();
    *out++ = '{';

    bool first = true;
    auto emit = [&](const char *name, const auto &value) {
      if (!first) {
        *out++ = ',';
        *out++ = ' ';
      }
      out = fmt::format_to(out, "{}={}", name, value);
      first = false;
    };

    if (info.name) {
      emit("name", *info.name);
    }
    emit("width", info.width);
    emit("height", info.height);
    emit("channels", info.channels);
    emit("storage", info.storage);
    emit("format", info.format);
    emit("type", info.type);

    *out++ = '}';
    return out;
  }
};

template <> struct fmt::formatter<denox::compiler::Tensor> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::compiler::Tensor &t, FormatContext &ctx) const {
    return fmt::format_to(ctx.out(), "{{size={}, alignment={}, info={}}}",
                          t.size, t.alignment, t.info);
  }
};
