#pragma once

#include "memory/container/span.hpp"
#include "memory/hypergraph/AdjGraph.hpp"
#include "memory/hypergraph/EdgeId.hpp"
#include "memory/hypergraph/NodeId.hpp"
#include "memory/hypergraph/NullWeight.hpp"
#include "memory/container/vector.hpp"

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

  explicit ConstGraph(const AdjGraph<V, E, W> &graph) {

    std::size_t nodeCount = graph.nodeCount();
    std::size_t edgeCount = graph.edgeCount();

    m_nodeData.reserve(nodeCount);
    m_nodes.reserve(nodeCount);
    m_edgeData.reserve(edgeCount);
    m_edges.reserve(edgeCount);

    // Pass 0: Compact IDs and build remapping tables.
    std::size_t maxNodeId{0};
    for (typename AdjGraph<V, E>::const_node_iterator::Node n :
         graph.nodes()) {
      maxNodeId = std::max(static_cast<std::size_t>(n.id()), maxNodeId);
    }
    std::size_t maxEdgeId{0};
    for (const typename AdjGraph<V, E>::const_edge_iterator::EdgeInfo &e :
         graph.edges()) {
      maxEdgeId = std::max(static_cast<std::size_t>(e.id()), maxEdgeId);
    }
    denox::memory::vector<NodeId> nodeRemap(maxNodeId + 1, NodeId{0});
    {
      std::size_t ix = 0;
      for (const typename AdjGraph<V, E>::const_node_iterator::Node &n :
           graph.nodes()) {
        nodeRemap[n.id()] = NodeId{ix++};
      }
    }
    denox::memory::vector<EdgeId> edgeRemap(maxEdgeId + 1, EdgeId{0});
    {
      std::size_t ix = 0;
      for (const typename AdjGraph<V, E>::const_edge_iterator::EdgeInfo &e :
           graph.edges()) {
        edgeRemap[e.id()] = EdgeId{ix++};
      }
    }

    // Pass 1: Count indeg and outdeg and srclen
    denox::memory::vector<unsigned int> indeg(nodeCount, 0);
    denox::memory::vector<unsigned int> outdeg(nodeCount, 0);
    denox::memory::vector<unsigned int> srclen(edgeCount, 0);

    for (const auto &e : graph.edges()) {
      std::size_t ne = edgeRemap[e.id()];
      auto srcs = e.edge().src();
      srclen[ne] = static_cast<unsigned int>(srcs.size());
      std::size_t dv = nodeRemap[e.edge().dst()];
      ++indeg[dv];

      for (NodeId u : srcs) {
        std::size_t nu = nodeRemap[u];
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
      for (const typename AdjGraph<V, E>::const_node_iterator::Node &n :
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
      for (const typename AdjGraph<V, E>::const_edge_iterator::EdgeInfo &e :
           graph.edges()) {
        m_edgeData.emplace_back(e.edge().payload());
        NodeId di{nodeRemap[e.edge().dst()]};
        m_edges.emplace_back(e.edge().weight(), srclenPrefix[idx],
                             srclenPrefix[idx] + srclen[idx], di);
        std::size_t nx = srclenPrefix[idx];
        for (const NodeId &src : e.edge().src()) {
          NodeId si{nodeRemap[src]};
          m_nodeIds[nx++] = si;
          m_edgeIds[outdegPrefix[si]++] = EdgeId{idx};
        }

        m_edgeIds[indegPrefix[di]++] = EdgeId{idx};
        ++idx;
      }
    }
  }

  denox::memory::span<const EdgeId> outgoing(NodeId node) const {
    return denox::memory::span{m_edgeIds.begin() + static_cast<std::ptrdiff_t>(m_nodes[node].outgoingBegin),
                     m_edgeIds.begin() + static_cast<std::ptrdiff_t>(m_nodes[node].outgoingEnd)};
  }

  denox::memory::span<const EdgeId> incoming(NodeId node) const {
    return denox::memory::span{m_edgeIds.begin() + m_nodes[node].incomingBegin,
                     m_edgeIds.begin() + m_nodes[node].incomingEnd};
  }

  const V &get(NodeId node) const { return m_nodeData[node]; }
  const E &get(EdgeId edge) const { return m_edgeData[edge]; }
  const W &weight(EdgeId edge) const { return m_edges[edge].weight; }

  denox::memory::span<const NodeId> src(EdgeId edge) const {
    return denox::memory::span{m_nodeIds.begin() + m_edges[edge].srcBegin,
                     m_nodeIds.begin() + m_edges[edge].srcEnd};
  }

  NodeId dst(EdgeId edge) const { return m_edges[edge].dst; }

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
