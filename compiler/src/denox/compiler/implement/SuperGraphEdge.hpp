#pragma once

#include "denox/compiler/implement/ComputeDispatch.hpp"
#include "denox/compiler/implement/MemoryConstrain.hpp"
#include "denox/compiler/implement/Parameter.hpp"
#include "denox/memory/container/vector.hpp"
#include <fmt/format.h>
#include <fmt/ranges.h>

namespace denox::compiler {

struct SuperGraphEdge {
  static constexpr size_t DISPATCH_SVO = 1;
  static constexpr size_t PARAM_SVO = 1;
  memory::small_vector<ComputeDispatch, DISPATCH_SVO> dispatches;
  memory::vector<MemoryImplicitConcatConstrain> memoryConstrains;
  memory::small_vector<Parameter, PARAM_SVO> parameters; // initalizers for tensors
};

} // namespace denox::compiler

template <> struct fmt::formatter<denox::compiler::SuperGraphEdge> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::compiler::SuperGraphEdge &e,
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

    if (!e.dispatches.empty())
      emit("dispatches", e.dispatches);

    if (!e.memoryConstrains.empty())
      emit("memoryConstrains", e.memoryConstrains);

    if (!e.parameters.empty())
      emit("parameters", e.parameters);

    *out++ = '}';
    return out;
  }
};
