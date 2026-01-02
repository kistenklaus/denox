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
  memory::ConstGraph<TensorId, SuperGraphEdge> graph;
  std::vector<Tensor> tensors;
  memory::small_vector<memory::NodeId, 2> inputs;
  memory::small_vector<memory::NodeId, 2> outputs;

  SymGraph symGraph;
};

} // namespace denox::compiler

template <> struct fmt::formatter<denox::compiler::SuperGraph> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::compiler::SuperGraph &sg, FormatContext &ctx) const {
    auto out = ctx.out();
    const auto &graph = sg.graph;

    fmt::format_to(out, "{{\n");
    fmt::format_to(out, "-inputs: [");
    {
      bool first = true;
      for (auto nid : sg.inputs) {
        if (!first) {
          fmt::format_to(out, ", ");
        }
        first = false;
        fmt::format_to(out, "T{}", graph.get(nid));
      }
    }
    fmt::format_to(out, "]\n");

    fmt::format_to(out, "-outputs: [");
    {
      bool first = true;
      for (auto nid : sg.outputs) {
        if (!first) {
          fmt::format_to(out, ", ");
        }
        first = false;
        fmt::format_to(out, "T{}", graph.get(nid));
      }
    }
    fmt::format_to(out, "]\n");

    fmt::format_to(out, "-tensors:\n");
    for (std::size_t i = 0; i < graph.nodeCount(); ++i) {
      denox::memory::NodeId nid{i};
      fmt::format_to(out, "  T{} {}\n", graph.get(nid).index,
                     sg.tensors[graph.get(nid).index]);
    }

    fmt::format_to(out, "-operations:\n");
    for (std::size_t i = 0; i < graph.edgeCount(); ++i) {
      denox::memory::EdgeId eid{i};

      // src list
      fmt::format_to(out, "  Op{}: [", i);
      bool first = true;
      for (auto src : graph.src(eid)) {
        if (!first) {
          fmt::format_to(out, ", ");
        }
        first = false;
        fmt::format_to(out, "T{}", graph.get(src).index);
      }
      fmt::format_to(out, "] -> [T{}]", graph.get(graph.dst(eid)).index);

      // payload
      fmt::format_to(out, " {}", graph.get(eid));

      fmt::format_to(out, "\n");
    }

    fmt::format_to(out, "}}");

    return out;
  }
};
