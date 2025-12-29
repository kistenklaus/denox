#pragma once

#include "denox/compiler/implement/MemoryConstrain.hpp"
#include "denox/compiler/implement/Parameter.hpp"
#include "denox/compiler/implement/SuperGraphEdge.hpp"
#include "denox/compiler/implement/Tensor.hpp"
#include "denox/compiler/specialization/TensorInstance.hpp"
#include "denox/memory/hypergraph/ConstGraph.hpp"
#include "denox/symbolic/SymGraph.hpp"

namespace denox::compiler {

struct SuperGraph {
  memory::ConstGraph<TensorInstance, SuperGraphEdge> graph;
  std::vector<Tensor> tensors;
  memory::small_vector<memory::NodeId, 2> inputs;
  memory::small_vector<memory::NodeId, 2> outputs;

  SymGraph symGraph;
};

} // namespace denox::compiler

template <> struct fmt::formatter<denox::compiler::SuperGraph> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::compiler::SuperGraph &sg, FormatContext &ctx) const {
    auto out = ctx.out();

    // ============================================================
    // Inputs
    // ============================================================

    fmt::format_to(out, "Inputs: [");
    {
      bool first = true;
      for (auto nid : sg.inputs) {
        if (!first)
          fmt::format_to(out, ", ");
        first = false;
        fmt::format_to(out, "N{}", *nid);
      }
    }
    fmt::format_to(out, "]\n");

    // ============================================================
    // Outputs
    // ============================================================

    fmt::format_to(out, "Outputs: [");
    {
      bool first = true;
      for (auto nid : sg.outputs) {
        if (!first)
          fmt::format_to(out, ", ");
        first = false;
        fmt::format_to(out, "N{}", *nid);
      }
    }
    fmt::format_to(out, "]\n");

    // ============================================================
    // Graph
    // ============================================================

    fmt::format_to(out, "Graph:\n");
    fmt::format_to(out, "{}", sg.graph);

    return out;
  }
};
