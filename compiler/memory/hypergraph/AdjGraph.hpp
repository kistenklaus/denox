#pragma once

#include "memory/container/optional.hpp"
#include "memory/container/span.hpp"
#include "memory/container/vector.hpp"
#include "memory/hypergraph/EdgeId.hpp"
#include "memory/hypergraph/NodeId.hpp"
#include "memory/hypergraph/NullWeight.hpp"

namespace denox::memory {

template <typename V, typename E, typename W = NullWeight> class AdjGraph {
public:
  NodeId addNode(V node) noexcept {
    if (m_nodeFreelist.empty()) {
      NodeId id{m_nodes.size()};
      m_nodes.push_back(node);
      return id;
    } else {
      NodeId id{m_nodeFreelist.back()};
      m_nodeFreelist.pop_back();
      m_nodes[id] = node;
      return id;
    }
  }

  template <typename... Args> NodeId emplaceNode(Args &&...args) noexcept {
    if (m_nodeFreelist.empty()) {
      NodeId id{m_nodes.size()};
      m_nodes.emplace_back();
      m_nodes.back().emplace(std::forward<Args>(args)...);
      return id;
    } else {
      NodeId id{m_nodeFreelist.back()};
      m_nodeFreelist.pop_back();
      m_nodes[id].emplace(std::forward<Args>(args)...);
      return id;
    }
  }

  // TODO we probably need some form of index pool.
  bool removeNode(NodeId node) noexcept {
    std::size_t id = node;
    assert(id < m_nodes.size());
    if (!m_nodes[id].has_value()) {
      return false;
    }
    m_nodes[id] = std::nullopt;
    m_nodeFreelist.push_back(id);
    return true;
  }

  EdgeId addEdge(NodeId src, NodeId dst, E edge, W weight = {}) noexcept {
    if (m_edgeFreelist.empty()) {
      EdgeId id{m_edges.size()};
      m_edges.push_back(Edge{src, dst, edge, weight});
      return id;
    } else {
      EdgeId id{m_edgeFreelist.back()};
      m_edgeFreelist.pop_back();
      m_edges[id].emplace(src, dst, edge, weight);
      return id;
    }
  }

  EdgeId addEdge(NodeId src0, NodeId src1, NodeId dst, E edge,
                 W weight = {}) noexcept {
    if (m_edgeFreelist.empty()) {
      EdgeId id{m_edges.size()};
      m_edges.push_back(
          denox::memory::optional<Edge>(Edge(src0, src1, dst, edge, weight)));
      return id;
    } else {
      EdgeId id{m_edgeFreelist.back()};
      m_edgeFreelist.pop_back();
      m_edges[id].emplace(src0, src1, dst, edge, weight);
      return id;
    }
  }

  EdgeId addEdge(denox::memory::span<const NodeId> src, NodeId dst, E edge,
                 W weight = {}) noexcept {
    if (m_edgeFreelist.empty()) {
      EdgeId id{m_edges.size()};
      m_edges.emplace_back(src, dst, edge, weight);
      return id;
    } else {
      EdgeId id{m_edgeFreelist.back()};
      m_edgeFreelist.pop_back();
      m_edges[id].emplace(src, dst, edge, weight);
      return id;
    }
  }

  bool removeEdge(EdgeId edge) noexcept {
    std::size_t id = edge;
    assert(id < m_edges.size());
    if (!m_edges[id].has_value()) {
      return false;
    }
    m_edges[id] = std::nullopt;
    m_edgeFreelist.push_back(id);
    return true;
  }

  const V &get(NodeId node) const noexcept { return *m_nodes[node]; }
  V &get(NodeId node) noexcept { return *m_nodes[node]; }
  const E &get(EdgeId edge) const noexcept { return m_edges[edge]->payload(); }
  E &get(EdgeId edge) noexcept { return m_edges[edge]->payload(); }
  const W &weight(EdgeId edge) const noexcept {
    return m_edges[edge]->weight();
  }
  W &weight(EdgeId edge) noexcept { return m_edges[edge]->weight(); }

  std::size_t nodeCount() const {
    return m_nodes.size() - m_nodeFreelist.size();
  }
  std::size_t edgeCount() const {
    return m_edges.size() - m_edgeFreelist.size();
  }

  struct const_node_iterator {
    struct Node {
      friend struct const_node_iterator;
      NodeId id() const { return NodeId{m_id}; }
      const V &node() const { return m_node->value(); }

    private:
      Node(NodeId id, const denox::memory::optional<V> *node)
          : m_id(id), m_node(node) {}
      std::uint64_t m_id;
      const denox::memory::optional<V> *m_node;
    };
    struct NodePtr {
      explicit NodePtr(Node node) : m_value(node) {}
      const Node *operator->() const { return m_value; }
      const Node &operator*() const { return m_value; }

    private:
      Node m_value;
    };
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = Node;
    using pointer = NodePtr;
    using reference = Node;
    explicit const_node_iterator(const denox::memory::optional<V> *nodes,
                                 std::size_t idx, std::size_t end)
        : m_current(NodeId{idx}, nodes), m_end(end) {}

    const_node_iterator() : m_current(NodeId{0}, nullptr), m_end(0) {}

    reference operator*() const { return m_current; }
    pointer operator->() const { return NodePtr(m_current); }

    const_node_iterator &operator++() {
      do {
        ++m_current.m_id;
        if (m_current.m_id == m_end) {
          m_current.m_node = nullptr;
          break;
        }
        ++m_current.m_node;
      } while (!m_current.m_node->has_value());
      return *this;
    }
    const_node_iterator operator++(int) {
      const_node_iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    friend bool operator==(const const_node_iterator &a,
                           const const_node_iterator &b) {
      return a.m_current.m_id == b.m_current.m_id;
    };
    friend bool operator!=(const const_node_iterator &a,
                           const const_node_iterator &b) {
      return a.m_current.m_id != b.m_current.m_id;
    };

  private:
    Node m_current;
    std::size_t m_end;
  };

  class Edge {
  public:
    explicit Edge(NodeId src, NodeId dst, E payload, W weight = {})
        : m_src({src}), m_dst(dst), m_payload(std::move(payload)),
          m_weight(std::move(weight)) {}
    explicit Edge(NodeId src0, NodeId src1, NodeId dst, E payload,
                  W weight = {})
        : m_src({src0, src1}), m_dst(dst), m_payload(std::move(payload)),
          m_weight(std::move(weight)) {}
    explicit Edge(denox::memory::span<const NodeId> src, NodeId dst, E payload,
                  W weight = {})
        : m_src(src.begin(), src.end()), m_dst(dst),
          m_payload(std::move(payload)), m_weight(std::move(weight)) {}

    denox::memory::span<const NodeId> src() const { return m_src; }
    NodeId dst() const { return m_dst; }

    const E &payload() const { return m_payload; }
    E &payload() { return m_payload; }
    const W &weight() const { return m_weight; }
    W &weight() { return m_weight; }

  private:
    denox::memory::vector<NodeId> m_src;
    NodeId m_dst;

    [[no_unique_address]] E m_payload;

    [[no_unique_address]] W m_weight;
  };

  struct const_edge_iterator {
    struct EdgeInfo {
      friend struct const_edge_iterator;
      EdgeId id() const { return EdgeId{m_id}; }
      const Edge &edge() const { return m_edge->value(); }

    private:
      EdgeInfo(EdgeId id, const denox::memory::optional<Edge> *edge)
          : m_id(id), m_edge(edge) {}

      std::uint64_t m_id;
      const denox::memory::optional<Edge> *m_edge;
    };
    struct EdgePtr {
      explicit EdgePtr(EdgeInfo edge) : m_value(edge) {}
      const EdgeInfo *operator->() const { return m_value; }
      const EdgeInfo &operator*() const { return m_value; }

    private:
      EdgeInfo m_value;
    };
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = EdgeInfo;
    using pointer = EdgePtr;
    using reference = EdgeInfo;
    explicit const_edge_iterator(const denox::memory::optional<Edge> *edge,
                                 std::size_t idx, std::size_t end)
        : m_current(EdgeId{idx}, edge), m_end(end) {}

    const_edge_iterator() : m_current(EdgeId{0}, nullptr), m_end(0) {}

    reference operator*() const { return m_current; }
    pointer operator->() const { return EdgePtr(m_current); }

    const_edge_iterator &operator++() {
      do {
        ++m_current.m_id;
        if (m_current.m_id == m_end) {
          m_current.m_edge = nullptr;
          break;
        }
        ++m_current.m_edge;
      } while (!m_current.m_edge->has_value());
      return *this;
    }
    const_edge_iterator operator++(int) {
      const_edge_iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    friend bool operator==(const const_edge_iterator &a,
                           const const_edge_iterator &b) {
      return a.m_current.m_id == b.m_current.m_id;
    };
    friend bool operator!=(const const_edge_iterator &a,
                           const const_edge_iterator &b) {
      return a.m_current.m_id != b.m_current.m_id;
    };

  private:
    EdgeInfo m_current;
    std::size_t m_end;
  };

  [[nodiscard]] std::ranges::subrange<const_node_iterator>
  nodes() const noexcept {
    auto it = std::ranges::find_if(m_nodes,
                                   [](const auto &x) { return x.has_value(); });
    if (it == m_nodes.end()) {
      return {const_node_iterator{}, const_node_iterator{}};
    }
    std::ptrdiff_t idx = std::distance(m_nodes.begin(), it);
    const_node_iterator begin{m_nodes.data() + idx, static_cast<std::size_t>(idx), m_nodes.size()};
    return {begin, const_node_iterator{m_nodes.data() + m_nodes.size(),
                                       m_nodes.size(), m_nodes.size()}};
  }

  [[nodiscard]] std::ranges::subrange<const_edge_iterator>
  edges() const noexcept {
    auto it = std::ranges::find_if(m_edges,
                                   [](const auto &x) { return x.has_value(); });
    if (it == m_edges.end()) {
      return {const_edge_iterator{}, const_edge_iterator{}};
    }
    std::ptrdiff_t idx = std::distance(m_edges.begin(), it);
    const_edge_iterator begin{m_edges.data() + idx, static_cast<std::size_t>(idx), m_edges.size()};
    return {begin, const_edge_iterator{m_edges.data() + m_edges.size(),
                                       m_edges.size(), m_edges.size()}};
  }

private:
  denox::memory::vector<denox::memory::optional<V>> m_nodes;
  denox::memory::vector<std::size_t> m_nodeFreelist;
  denox::memory::vector<denox::memory::optional<Edge>> m_edges;
  denox::memory::vector<std::size_t> m_edgeFreelist;
};

} // namespace denox::memory
