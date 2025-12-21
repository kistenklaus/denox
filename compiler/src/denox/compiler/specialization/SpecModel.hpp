#pragma once

#include "denox/compiler/frontend/model/ComputeOp.hpp"
#include "denox/compiler/specialization/TensorInstance.hpp"
#include "denox/memory/hypergraph/LinkedGraph.hpp"
#include <fmt/core.h>
#include <unordered_map>
#include <vector>

namespace denox::compiler {

struct SpecModel {
  using Graph = memory::LinkedGraph<TensorInstance, ComputeOp>;
  Graph graph;

  std::vector<Graph::NodeHandle> inputs;
  std::vector<Graph::NodeHandle> outputs;
};

} // namespace denox::compiler

template <> struct fmt::formatter<denox::compiler::SpecModel> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::compiler::SpecModel &model,
              FormatContext &ctx) const {
    using SpecModel = denox::compiler::SpecModel;
    using Graph = SpecModel::Graph;
    using Node = Graph::Node;
    using Edge = Graph::Edge;

    auto out = ctx.out();

    // ============================================================
    // Inputs / Outputs
    // ============================================================

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
    fmt::format_to(out, "]\n\n");

    // ============================================================
    // Collect nodes and edges
    // ============================================================

    std::vector<const Node *> nodes;
    std::unordered_map<const Edge *, std::size_t> edgeIds;
    std::vector<const Edge *> edges;

    auto get_edge_id = [&](const Edge *e) -> std::size_t {
      auto it = edgeIds.find(e);
      if (it != edgeIds.end())
        return it->second;
      std::size_t id = edges.size();
      edgeIds.emplace(e, id);
      edges.push_back(e);
      return id;
    };

    // Discover nodes by walking reachable graph
    std::unordered_map<const Node *, bool> visited;

    std::vector<const Node *> stack;
    for (const auto &in : model.inputs)
      stack.push_back(&*in);

    while (!stack.empty()) {
      const Node *n = stack.back();
      stack.pop_back();

      if (visited[n])
        continue;
      visited[n] = true;

      nodes.push_back(n);

      for (const Edge &e : n->outgoing()) {
        get_edge_id(&e);
        stack.push_back(&e.dst());
        for (const auto &src : e.srcs())
          stack.push_back(&src);
      }
    }

    // ============================================================
    // Nodes
    // ============================================================

    fmt::format_to(out, "Nodes:\n");

    for (const Node *n : nodes) {
      fmt::format_to(out, "  N{} {{ value={} }}\n", *n->id(), n->value());
    }

    // ============================================================
    // Edges
    // ============================================================

    fmt::format_to(out, "\nEdges:\n");

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

      fmt::format_to(out, "] -> N{}  {}\n", *e->dst().id(), e->value());
    }

    return out;
  }
};
