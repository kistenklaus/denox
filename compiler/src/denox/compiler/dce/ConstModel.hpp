#pragma once

#include "denox/common/ComputeOp.hpp"
#include "denox/compiler/specialization/TensorInstance.hpp"
#include "denox/memory/hypergraph/ConstGraph.hpp"
#include <fmt/core.h>

namespace denox::compiler {

struct ConstModel {
  using Graph = memory::ConstGraph<TensorInstance, ComputeOp>;
  Graph graph;
  std::vector<memory::NodeId> inputs;
  std::vector<memory::NodeId> outputs;
};

} // namespace denox::compiler

namespace fmt {

template <> struct formatter<denox::compiler::ConstModel> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::compiler::ConstModel &model,
              FormatContext &ctx) const {
    auto out = ctx.out();

    // Inputs
    fmt::format_to(out, "Inputs: [");
    bool first = true;
    for (const auto &id : model.inputs) {
      if (!first)
        fmt::format_to(out, ", ");
      first = false;
      fmt::format_to(out, "N{}", *id);
    }
    fmt::format_to(out, "]\n");

    // Outputs
    fmt::format_to(out, "Outputs: [");
    first = true;
    for (const auto &id : model.outputs) {
      if (!first)
        fmt::format_to(out, ", ");
      first = false;
      fmt::format_to(out, "N{}", *id);
    }
    fmt::format_to(out, "]\n");

    // Graph
    fmt::format_to(out, "{}", model.graph);

    return out;
  }
};

} // namespace fmt
