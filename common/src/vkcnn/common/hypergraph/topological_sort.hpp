#pragma once

#include "vkcnn/common/hypergraph/ConstGraph.hpp"
#include "vkcnn/common/hypergraph/NodeId.hpp"
#include <queue>
#include <stdexcept>
#include <vector>
namespace vkcnn::hypergraph {

template <typename V, typename E>
std::vector<NodeId> topologicalSort(const ConstGraph<V, E> &hypergraph) {

  std::size_t nodeCount = hypergraph.nodeCount();
  std::size_t edgeCount = hypergraph.edgeCount();

  std::vector<unsigned int> edgeRemainingSrc(edgeCount, 0);
  std::vector<unsigned int> nodeUnsatisfiedIn(nodeCount, 0);

  for (std::size_t ei = 0; ei < edgeCount; ++ei) {
    EdgeId e{ei};
    auto srcs = hypergraph.src(e);
    edgeRemainingSrc[ei] = static_cast<unsigned int>(srcs.size());
    NodeId d = hypergraph.dst(e);
    nodeUnsatisfiedIn[d]++;
  }

  std::queue<NodeId> ready;
  for (std::size_t ni = 0; ni < nodeCount; ++ni) {
    if (nodeUnsatisfiedIn[ni] == 0) {
      ready.push(NodeId{ni});
    }
  }

  std::vector<NodeId> order;
  order.reserve(nodeCount);

  while (!ready.empty()) {
    NodeId n = ready.front();
    ready.pop();
    order.push_back(n);

    for (EdgeId e : hypergraph.outgoing(n)) {
      if (edgeRemainingSrc[e] > 0 && hypergraph.src(e).size() > 0) {

        if (--edgeRemainingSrc[e] == 0) {
          NodeId d = hypergraph.dst(e);
          if (--nodeUnsatisfiedIn[d] == 0) {
            ready.push(d);
          }
        }
      }
    }
  }
  if (order.size() != nodeCount) {
    throw std::runtime_error(
        "topologicialSort: raph contains a cycle or a dangling dependency");
  }

  return order;
}

} // namespace vkcnn::hypergraph
