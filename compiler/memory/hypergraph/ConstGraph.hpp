#pragma once

#include "memory/container/dynamic_bitset.hpp"
#include "memory/container/hashmap.hpp"
#include "memory/container/small_vector.hpp"
#include "memory/container/span.hpp"
#include "memory/container/vector.hpp"
#include "memory/hypergraph/AdjGraph.hpp"
#include "memory/hypergraph/EdgeId.hpp"
#include "memory/hypergraph/LinkedGraph.hpp"
#include "memory/hypergraph/NodeId.hpp"
#include "memory/hypergraph/NullWeight.hpp"
#include <tuple>

namespace denox::memory {

template <typename V, typename E, typename W = NullWeight> class ConstGraph {
public:
  struct Node {
    std::size_t incomingBegin;
    std::size_t incomingEnd;
    std::size_t outgoingBegin;
    std::size_t outgoingEnd;
  };

  struct Edge {
    W weight;
    std::size_t srcBegin;
    std::size_t srcEnd;
    NodeId dst;
  };

  // NOTE: Constructs a ConstGraph, from a
  // LinkedGraph input and output node.
  // A node X, in only included in the ConstGraph, iff. all of the following:
  // - There exists a path from input to X.
  // - There exists a path from X to output.
  // A edge is only included iff. there exists
  // a path from input to output which includes the edge.
  //
  // Returns the NodeId of the input and output,
  // within the constructured ConstGraph.
  //
  template <typename Allocator = mallocator>
  static std::tuple<memory::vector<NodeId>, ConstGraph>
  from(const LinkedGraph<V, E, W, Allocator>::NodeHandle &input,
       const LinkedGraph<V, E, W, Allocator>::NodeHandle &output) {
    using LinkedGraph = LinkedGraph<V, E, W, Allocator>;
    using AdjGraph = AdjGraph<V, E, W>;
    AdjGraph adj;
    using NodeHandle = LinkedGraph::NodeHandle;

    std::size_t upperNodeCount = input.upperNodeCount();
    assert(output.upperNodeCount() == upperNodeCount);

    memory::dynamic_bitset visited(upperNodeCount);
    memory::vector<NodeHandle> stack;
    stack.reserve(upperNodeCount);

    memory::vector<memory::NodeId> adjNodes(upperNodeCount);
    adjNodes[output->id()] = adj.addNode(output->value());

    stack.push_back(output);

    while (!stack.empty()) {
      NodeHandle node = stack.back();
      stack.pop_back();
      memory::NodeId id = node->id();
      if (visited[*id]) {
        continue;
      }
      visited[*id] = true;

      for (const auto &edge : node->incoming()) {
        memory::small_vector<memory::NodeId, 2> srcs;
        for (auto &src : edge.srcs()) {
          if (visited[src.id()]) {
            memory::NodeId adjSrc = adjNodes[src.id()];
            srcs.push_back(adjSrc);
          } else {
            memory::NodeId adjSrc = adj.addNode(src.value());
            adjNodes[src.id()] = adjSrc;
            srcs.push_back(adjSrc);
            stack.push_back(NodeHandle(src));
          }
        }
        adj.addEdge(std::span<const memory::NodeId>(srcs.begin(), srcs.end()),
                    adjNodes[*id], edge.value(), memory::NullWeight{});
      }
    }
    ConstGraph<V, E, W> graph{adj};
    return std::make_tuple(adjNodes, std::move(adj));
  }

  explicit ConstGraph(const AdjGraph<V, E, W> &graph) {

    std::size_t nodeCount = graph.nodeCount();
    std::size_t edgeCount = graph.edgeCount();

    m_nodeData.reserve(nodeCount);
    m_nodes.reserve(nodeCount);
    m_edgeData.reserve(edgeCount);
    m_edges.reserve(edgeCount);

    // Pass 0: Compact IDs and build remapping tables.
    std::size_t maxNodeId{0};
    for (typename AdjGraph<V, E, W>::const_node_iterator::Node n : graph.nodes()) {
      maxNodeId = std::max(static_cast<std::uint64_t>(n.id()), static_cast<std::uint64_t>(maxNodeId));
    }
    std::size_t maxEdgeId{0};
    for (const typename AdjGraph<V, E, W>::const_edge_iterator::EdgeInfo &e :
         graph.edges()) {
      maxEdgeId = std::max(static_cast<std::uint64_t>(e.id()), static_cast<std::uint64_t>(maxEdgeId));
    }
    denox::memory::vector<NodeId> nodeRemap(maxNodeId + 1, NodeId{0});
    {
      std::size_t ix = 0;
      for (const typename AdjGraph<V, E, W>::const_node_iterator::Node &n :
           graph.nodes()) {
        nodeRemap[*n.id()] = NodeId{ix++};
      }
    }
    denox::memory::vector<EdgeId> edgeRemap(maxEdgeId + 1, EdgeId{0});
    {
      std::size_t ix = 0;
      for (const typename AdjGraph<V, E, W>::const_edge_iterator::EdgeInfo &e :
           graph.edges()) {
        edgeRemap[*e.id()] = EdgeId{ix++};
      }
    }

    // Pass 1: Count indeg and outdeg and srclen
    denox::memory::vector<unsigned int> indeg(nodeCount, 0);
    denox::memory::vector<unsigned int> outdeg(nodeCount, 0);
    denox::memory::vector<unsigned int> srclen(edgeCount, 0);

    for (const auto &e : graph.edges()) {
      std::size_t ne = *edgeRemap[*e.id()];
      auto srcs = e.edge().src();
      srclen[ne] = static_cast<unsigned int>(srcs.size());
      std::size_t dv = *nodeRemap[*e.edge().dst()];
      ++indeg[dv];

      for (NodeId u : srcs) {
        std::size_t nu = *nodeRemap[*u];
        ++outdeg[nu];
      }
    }
    // Pass 2 : Compute prefix sums over indeg, outdeg and srclen
    denox::memory::vector<std::size_t> indegPrefix(nodeCount + 1, 0);
    denox::memory::vector<std::size_t> outdegPrefix(nodeCount + 1, 0);
    denox::memory::vector<std::size_t> srclenPrefix(edgeCount + 1, 0);
    for (std::size_t v = 0; v < nodeCount; ++v) {
      indegPrefix[v + 1] = indegPrefix[v] + indeg[v];
    }
    for (std::size_t v = 0; v < nodeCount; ++v) {
      outdegPrefix[v + 1] = outdegPrefix[v] + outdeg[v];
    }
    for (std::size_t e = 0; e < edgeCount; ++e) {
      srclenPrefix[e + 1] = srclenPrefix[e] + srclen[e];
    }

    m_edgeIds.resize(indegPrefix.back() + outdegPrefix.back(), EdgeId{0});
    m_nodeIds.resize(srclenPrefix.back(), NodeId{0});

    for (std::size_t v = 0; v < nodeCount; ++v) {
      outdegPrefix[v] = indegPrefix.back() + outdegPrefix[v];
    }

    // Pass 3: Copy node data + metadata & edge data + metadata.
    //         Populate id arrays.
    {
      std::size_t idx = 0;
      for (const typename AdjGraph<V, E, W>::const_node_iterator::Node &n :
           graph.nodes()) {
        m_nodeData.emplace_back(n.node());
        m_nodes.emplace_back(indegPrefix[idx], indegPrefix[idx] + indeg[idx],
                             outdegPrefix[idx],
                             outdegPrefix[idx] + outdeg[idx]);
        ++idx;
      }
    }
    {
      std::size_t idx = 0;
      for (const typename AdjGraph<V, E, W>::const_edge_iterator::EdgeInfo &e :
           graph.edges()) {
        m_edgeData.emplace_back(e.edge().payload());
        NodeId di{nodeRemap[*e.edge().dst()]};
        m_edges.emplace_back(e.edge().weight(), srclenPrefix[idx],
                             srclenPrefix[idx] + srclen[idx], di);
        std::size_t nx = srclenPrefix[idx];
        for (const NodeId &src : e.edge().src()) {
          NodeId si{nodeRemap[*src]};
          m_nodeIds[nx++] = si;
          m_edgeIds[outdegPrefix[*si]++] = EdgeId{idx};
        }

        m_edgeIds[indegPrefix[*di]++] = EdgeId{idx};
        ++idx;
      }
    }
  }

  denox::memory::span<const EdgeId> outgoing(NodeId node) const {
    return denox::memory::span<const EdgeId>{
        m_edgeIds.begin() +
            static_cast<std::ptrdiff_t>(m_nodes[*node].outgoingBegin),
        m_edgeIds.begin() +
            static_cast<std::ptrdiff_t>(m_nodes[*node].outgoingEnd)};
  }

  denox::memory::span<const EdgeId> incoming(NodeId node) const {
    return denox::memory::span<const EdgeId>{
        m_edgeIds.begin() +
            static_cast<std::ptrdiff_t>(m_nodes[*node].incomingBegin),
        m_edgeIds.begin() +
            static_cast<std::ptrdiff_t>(m_nodes[*node].incomingEnd)};
  }

  const V &get(NodeId node) const { return m_nodeData[*node]; }
  const E &get(EdgeId edge) const { return m_edgeData[*edge]; }
  const W &weight(EdgeId edge) const { return m_edges[*edge].weight; }

  denox::memory::span<const NodeId> src(EdgeId edge) const {
    return denox::memory::span<const NodeId>{
        m_nodeIds.begin() + static_cast<std::ptrdiff_t>(m_edges[*edge].srcBegin),
        m_nodeIds.begin() + static_cast<std::ptrdiff_t>(m_edges[*edge].srcEnd)};
  }

  NodeId dst(EdgeId edge) const { return m_edges[*edge].dst; }

  std::size_t nodeCount() const { return m_nodes.size(); }
  std::size_t edgeCount() const { return m_edges.size(); }

private:
  denox::memory::vector<NodeId> m_nodeIds;
  denox::memory::vector<EdgeId> m_edgeIds;
  denox::memory::vector<Node> m_nodes;
  denox::memory::vector<Edge> m_edges;
  denox::memory::vector<V> m_nodeData;
  denox::memory::vector<E> m_edgeData;
};
} // namespace denox::memory
