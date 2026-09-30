#pragma once

#include "all_minimum_cost_subgraph.hpp" // Shared deterministic branchy fixtures.
#include "denox/algorithm/minimum_cost_subgraph.hpp"

namespace denox_bench {
template <typename Run>
void benchmarkSingleMinimum(benchmark::State &state, Run run) {
  const unsigned width = static_cast<unsigned>(state.range(0));
  const auto fixture = makeBranchyCase(width, 0xB12A, false);
  Builder builder;
  for (unsigned v = 0; v < 2 * width + 1; ++v)
    builder.addNode(v);
  for (unsigned e = 0; e < fixture.edges.size(); ++e) {
    const auto &edge = fixture.edges[e];
    std::vector<NodeId> sources;
    for (auto v : edge.sources)
      sources.emplace_back(v);
    builder.addEdge(sources, NodeId{edge.destination}, e, edge.cost);
  }
  const std::vector<NodeId> inputs{NodeId{0}};
  std::vector<NodeId> branches;
  for (auto v : fixture.outputs)
    branches.emplace_back(v);
  const NodeId output = builder.addNode(2 * width + 1);
  builder.addEdge(branches, output, fixture.edges.size(), 1);
  const std::vector<NodeId> outputs{output};
  const Graph graph(builder);
  for (auto _ : state) {
    auto result = run(graph, inputs, outputs);
    benchmark::DoNotOptimize(result);
    benchmark::ClobberMemory();
  }
  state.counters["nodes"] = graph.nodeCount();
  state.counters["edges"] = graph.edgeCount();
  state.counters["outputs"] = outputs.size();
  state.counters["alternatives"] = 8;
  state.counters["alternative_rank"] = 3;
  state.counters["join_rank"] = width;
}

static void minimumCostSubgraphDenseHyperedgeExact(benchmark::State &state) {
  benchmarkSingleMinimum(
      state, [](const Graph &graph, const auto &in, const auto &out) {
        return denox::algorithm::minimum_cost_subgraph(graph, in, out, 0);
      });
}
static void minimumCostSubgraphDenseHyperedgeBeam64(benchmark::State &state) {
  benchmarkSingleMinimum(
      state, [](const Graph &graph, const auto &in, const auto &out) {
        return denox::algorithm::minimum_cost_subgraph(graph, in, out, 64);
      });
}
BENCHMARK(minimumCostSubgraphDenseHyperedgeExact)
    ->Arg(10)
    ->Arg(14)
    ->Arg(18)
    ->Arg(22)
    ->Arg(24)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(minimumCostSubgraphDenseHyperedgeBeam64)
    ->Arg(10)
    ->Arg(14)
    ->Arg(18)
    ->Arg(22)
    ->Arg(24)
    ->Arg(30)
    ->Arg(36)
    ->Arg(42)
    ->Arg(48)
    ->Unit(benchmark::kMillisecond);
} // namespace denox_bench
