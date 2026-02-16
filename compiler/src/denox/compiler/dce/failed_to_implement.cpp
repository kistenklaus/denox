#include "denox/compiler/dce/failed_to_implement.hpp"
#include "denox/algorithm/minimum_const_subgraph.hpp"
#include "denox/compiler/dce/ConstModel.hpp"
#include <stdexcept>

void denox::compiler::failed_to_implement(const SuperGraph &supergraph,
                                          const ConstModel &model) {

  // ===== Explaination of the algorithmic idea ========
  // Problem:
  // Given a graph G with nodes V and edges E and additional edges E'.
  // Find the minimum amount of edges from E' to add to E, such that there
  // exists a hyperpath from all inputs nodes to all output nodes.
  //
  // Algorithm: Define graph G=(V, E & E')
  // With cost function c(e) = 0, if e \in E otherwise c(e) = 1.
  // Then find minimum-cost-subgraph of G.
  // All edges of the minimum-cost-subgraph, which have cost c(e) = 1, are
  // part of the minimum edge set from E' to add to E such that there exists
  // a path.
  //
  // Mapping: E are edges of the supergraph and E' are edges of the model.
  // Both supergraph and model have the same nodes, therefor this works and in
  // my opinion is beautifully simple. Although probably not optimal for error
  // message this is more than enough!

  assert(supergraph.graph.nodeCount() == model.graph.nodeCount());
  const size_t N = supergraph.graph.nodeCount();

  struct Phantom {};

  memory::AdjGraph<Phantom, memory::EdgeId, uint32_t> agraph;
  for (uint32_t n = 0; n < N; ++n) {
    agraph.addNode({});
  }

  uint32_t M = static_cast<uint32_t>(supergraph.graph.edgeCount());
  for (uint32_t e = 0; e < M; ++e) {
    memory::EdgeId eid{e};
    agraph.addEdge(supergraph.graph.src(eid), supergraph.graph.dst(eid),
                   memory::EdgeId{}, 0);
  }

  uint32_t K = static_cast<uint32_t>(model.graph.edgeCount());
  for (uint32_t e = 0; e < K; ++e) {
    memory::EdgeId eid{e};
    agraph.addEdge(model.graph.src(eid), model.graph.dst(eid), eid, 1);
  }

  memory::ConstGraph<Phantom, memory::EdgeId, uint32_t> graph{
      std::move(agraph)};
  memory::AdjGraph<Phantom, memory::EdgeId, uint32_t> aminimum_cost_subgraph =
      algorithm::minimum_cost_subgraph(graph, supergraph.inputs,
                                       supergraph.outputs);
  memory::ConstGraph<Phantom, memory::EdgeId, uint32_t> minimum_cost_subgraph{
      std::move(aminimum_cost_subgraph)};

  memory::vector<memory::EdgeId> unimplemented_ops;

  for (uint32_t e = 0; e < minimum_cost_subgraph.edgeCount(); ++e) {
    memory::EdgeId eid{e};
    if (minimum_cost_subgraph.weight(eid) == 1) {
      memory::EdgeId opid = minimum_cost_subgraph.get(eid);
      assert(opid != memory::EdgeId{});
      unimplemented_ops.push_back(opid);
    }
  }

  std::string msg;
  for (memory::EdgeId eid : unimplemented_ops) {
    const ComputeOp &op = model.graph.get(eid);
    const auto srcsIds = model.graph.src(eid);
    const auto dstId = model.graph.dst(eid);

    std::string opString;
    if (srcsIds.size() == 1) {

    } else {
      assert(srcsIds.size() > 1);
    }

    msg += fmt::format("{}\n", op);
  }

  throw std::runtime_error(fmt::format("Failed to implement model:\n{}", msg));
}
