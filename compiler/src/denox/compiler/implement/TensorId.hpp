#pragma once

#include <cstdint>
#include <fmt/format.h>
#include <limits>
namespace denox::compiler {

struct TensorId {
  static constexpr std::uint64_t nullindex =
      std::numeric_limits<std::uint64_t>::max();
  std::uint64_t index = nullindex;
};

} // namespace denox::compiler

template <> struct fmt::formatter<denox::compiler::TensorId> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::compiler::TensorId &id, FormatContext &ctx) const {
    return fmt::format_to(ctx.out(), "{}", id.index);
  }
};
