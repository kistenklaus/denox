#include "denox/compiler/dce/prune_dead_supergraph.hpp"
#include "denox/compiler/implement/TensorId.hpp"
#include "denox/memory/container/dynamic_bitset.hpp"
#include "denox/memory/container/small_vector.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/memory/hypergraph/AdjGraph.hpp"
#include "denox/memory/hypergraph/ConstGraph.hpp"
#include <fmt/base.h>
#include <stdexcept>
#include <utility>

void denox::compiler::prune_dead_supergraph(SuperGraph &supergraph) {

  auto &graph = supergraph.graph;
  const size_t N = graph.nodeCount();
  const size_t M = graph.edgeCount();

  memory::vector<bool> reachable(N, true);
  while (true) {
    // 1. Forward reachability:
    memory::dynamic_bitset forwardReachable(N, false);
    {
      memory::vector<memory::NodeId> stack;
      stack.reserve(N);
      for (memory::NodeId input : supergraph.inputs) {
        stack.push_back(input);
      }
      while (!stack.empty()) {
        memory::NodeId nid = stack.back();
        stack.pop_back();
        if (forwardReachable[*nid]) {
          continue;
        }
        forwardReachable[*nid] = true;
        for (memory::EdgeId eid : graph.outgoing(nid)) {
          bool srcsReachable = true;
          for (memory::NodeId src : graph.src(eid)) {
            if (!reachable[*src]) {
              srcsReachable = false;
              break;
            }
          }
          if (!srcsReachable) {
            continue;
          }

          memory::NodeId dst = graph.dst(eid);
          if (!forwardReachable[*dst]) {
            stack.push_back(dst);
          }
        }
      }
    }

    // 2. Backwards reachability
    memory::dynamic_bitset backwardReachable(N, false);
    {
      memory::vector<memory::NodeId> stack;
      stack.reserve(N);
      for (memory::NodeId output : supergraph.outputs) {
        stack.push_back(output);
      }
      while (!stack.empty()) {
        memory::NodeId nid = stack.back();
        stack.pop_back();

        if (backwardReachable[*nid])
          continue;

        backwardReachable[*nid] = true;

        for (memory::EdgeId eid : graph.incoming(nid)) {
          bool srcsReachable = true;
          for (memory::NodeId src : graph.src(eid)) {
            if (!reachable[*src]) {
              srcsReachable = false;
              break;
            }
          }
          if (!srcsReachable) {
            continue;
          }

          for (memory::NodeId src : graph.src(eid)) {
            if (!backwardReachable[*src]) {
              stack.push_back(src);
            }
          }
        }
      }
    }
    size_t pruned = 0;
    for (uint32_t i = 0; i < N; ++i) {
      if (reachable[i] && !(forwardReachable[i] && backwardReachable[i])) {
        reachable[i] = false;
        pruned++;
      }
    }
    if (pruned == 0) {
      break;
    }
  }

  memory::AdjGraph<TensorId, SuperGraphEdge> pruned;
  // add surviving nodes

  memory::vector<memory::NodeId> nodeRemap(N);
  for (uint64_t i = 0; i < N; ++i) {
    memory::NodeId nid{i};
    if (reachable[*nid]) {
      memory::NodeId nnid = pruned.addNode(std::move(graph.get_rvalue(nid)));
      nodeRemap[*nid] = nnid;
    }
  }

  for (uint64_t i = 0; i < M; ++i) {
    memory::EdgeId eid{i};
    const memory::NodeId dst = graph.dst(eid);
    if (!reachable[*dst]) {
      continue; // pruned
    }
    bool allSrcAreLive = true;
    for (memory::NodeId nid : graph.src(eid)) {
      if (!reachable[*nid]) {
        allSrcAreLive = false;
        break;
      }
    }
    if (allSrcAreLive) {
      memory::small_vector<memory::NodeId, 2> srcs;
      for (const auto &src : graph.src(eid)) {
        srcs.push_back(nodeRemap[*src]);
      }
      pruned.addEdge(srcs, nodeRemap[*dst], std::move(graph.get_rvalue(eid)));
    }
  }

  memory::ConstGraph<TensorId, SuperGraphEdge> newGraph(std::move(pruned));

  supergraph.graph = std::move(newGraph);

  // remap inputs
  for (auto &nid : supergraph.inputs) {
    assert(nid != memory::NodeId{});
    nid = nodeRemap[*nid];
    if (!nid) {
      throw std::runtime_error("Failed to implement model");
    }
  }
  for (auto &nid : supergraph.outputs) {
    assert(nid != memory::NodeId{});
    nid = nodeRemap[*nid];
    assert(nid != memory::NodeId{});
  }

  for (uint32_t i = 0; i < supergraph.graph.nodeCount(); ++i) {
    memory::NodeId nid{i};
    if (std::ranges::find(supergraph.inputs, nid) == supergraph.inputs.end()) {
      assert(!supergraph.graph.incoming(nid).empty());
    }
  }
}
