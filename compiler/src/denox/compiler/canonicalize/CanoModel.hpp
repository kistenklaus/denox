#pragma once

#include "denox/compiler/frontend/model/ComputeOp.hpp"
#include "denox/compiler/frontend/model/ComputeTensor.hpp"
#include "denox/memory/hypergraph/LinkedGraph.hpp"
#include "denox/symbolic/SymGraph.hpp"
#include <fmt/core.h>
#include <fmt/format.h>

namespace denox::compiler {

struct CanoModel {
  using Graph = memory::LinkedGraph<ComputeTensor, ComputeOp>;
  Graph graph;
  std::vector<Graph::NodeHandle> inputs;
  std::vector<Graph::NodeHandle> outputs;
  SymGraph symGraph;
};

} // namespace denox::compiler

template <> struct fmt::formatter<denox::compiler::CanoModel> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::compiler::CanoModel &model,
              FormatContext &ctx) const {

    auto out = ctx.out();
    using Graph = denox::compiler::CanoModel::Graph;
    using Node = typename Graph::Node;
    using Edge = typename Graph::Edge;

    // ------------------------------------------------------------
    // Print inputs / outputs
    // ------------------------------------------------------------

    fmt::format_to(out, "Inputs: [");
    for (std::size_t i = 0; i < model.inputs.size(); ++i) {
      if (i)
        fmt::format_to(out, ", ");
      fmt::format_to(out, "N{}", *model.inputs[i]->id());
    }
    fmt::format_to(out, "]\n");

    fmt::format_to(out, "Outputs: [");
    for (std::size_t i = 0; i < model.outputs.size(); ++i) {
      if (i)
        fmt::format_to(out, ", ");
      fmt::format_to(out, "N{}", *model.outputs[i]->id());
    }
    fmt::format_to(out, "]\n");

    // ------------------------------------------------------------
    // Graph traversal
    // ------------------------------------------------------------

    std::vector<const Node *> nodes;
    std::unordered_set<const Node *> visited;

    std::vector<const Edge *> edges;
    std::unordered_map<const Edge *, std::size_t> edge_ids;

    auto get_edge_id = [&](const Edge *e) {
      auto it = edge_ids.find(e);
      if (it != edge_ids.end())
        return it->second;
      std::size_t id = edges.size();
      edge_ids.emplace(e, id);
      edges.push_back(e);
      return id;
    };

    std::function<void(const Node *)> visit;
    visit = [&](const Node *n) {
      if (!visited.insert(n).second)
        return;

      nodes.push_back(n);

      for (const Edge &e : n->incoming()) {
        get_edge_id(&e);
        visit(&e.dst());
        for (const auto &src : e.srcs())
          visit(&src);
      }

      for (const Edge &e : n->outgoing()) {
        get_edge_id(&e);
        visit(&e.dst());
        for (const auto &src : e.srcs())
          visit(&src);
      }
    };

    // ------------------------------------------------------------
    // Seed traversal from *known live nodes*
    // ------------------------------------------------------------

    for (const auto &h : model.inputs)
      visit(&*h);

    for (const auto &h : model.outputs)
      visit(&*h);

    // ------------------------------------------------------------
    // Print graph
    // ------------------------------------------------------------

    fmt::format_to(out, "Nodes:\n");
    for (const Node *n : nodes) {
      fmt::format_to(out, "  N{} {{ value={} }}\n", *n->id(), n->value());
    }
    fmt::format_to(out, "Edges:\n");

    for (std::size_t i = 0; i < edges.size(); ++i) {
      const Edge *e = edges[i];

      fmt::format_to(out, "  E{}: [", i);

      bool first = true;
      for (const auto &src : e->srcs()) {
        if (!first)
          fmt::format_to(out, ", ");
        first = false;
        fmt::format_to(out, "N{}", *src.id());
      }

      fmt::format_to(out, "] -> N{} {{ op={} }}\n", *e->dst().id(),
                     e->value() // ComputeOp formatter kicks in here
      );
    }

    return out;
  }
};
