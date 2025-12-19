#pragma once

#include "denox/algorithm/pattern_matching/EdgePattern.fwd.hpp"
#include "denox/algorithm/pattern_matching/NodePattern.fwd.hpp"
#include "denox/memory/allocator/mallocator.hpp"
#include "denox/memory/hypergraph/LinkedGraph.hpp"
#include "denox/memory/hypergraph/NullWeight.hpp"
#include <vector>
namespace denox::algorithm {

namespace pattern_matching::details {
template <typename V, typename E, typename W = memory::NullWeight,
          typename Allocator = memory::mallocator>
struct EdgeMatchControl {
  using LinkedGraph = memory::LinkedGraph<V, E, W, Allocator>;
  using NodeHandle = LinkedGraph::NodeHandle;

  EdgeMatchControl(NodeHandle node, LinkedGraph::EdgeIt it)
      : m_dirty(false), m_node(std::move(node)), m_iterator(it), m_next(it) {}

  void erase() {
    assert(m_dirty == false);
    m_dirty = true;
    auto outgoing = m_node->outgoing();
    m_next = outgoing.erase(m_iterator);
  }

  LinkedGraph::EdgeIt iterator() const { return m_iterator; }
  NodeHandle sourceNode() const { return m_node; }

  LinkedGraph::EdgeIt nextIterator() const {
    if (m_dirty) {
      return m_next;
    } else {
      typename LinkedGraph::EdgeIt next = m_iterator;
      return ++next;
    }
  }
  bool dirty() const { return m_dirty; }

private:
  bool m_dirty;
  NodeHandle m_node;
  LinkedGraph::EdgeIt m_iterator;
  LinkedGraph::EdgeIt m_next;
};
} // namespace pattern_matching::details

template <typename V, typename E, typename W, typename Allocator>
class LinkedGraphMatch;

template <typename V, typename E, typename W = memory::NullWeight,
          typename Allocator = memory::mallocator>
struct EdgeMatch {
  using CB = pattern_matching::details::EdgeMatchControl<V, E, W, Allocator>;
  using LinkedGraph = memory::LinkedGraph<V, E, W, Allocator>;
  using NodeHandle = LinkedGraph::NodeHandle;
  friend LinkedGraphMatch<V, E, W, Allocator>;
  EdgeMatch() : m_cb(nullptr) {}
  EdgeMatch(CB *cb) : m_cb(cb) {}

  void erase() {
    assert(m_cb != nullptr);
    return m_cb->erase();
  }

  LinkedGraph::EdgeIt outgoingIterator() const {
    assert(m_cb != nullptr);
    return m_cb->iterator();
  }
  LinkedGraph::EdgeIt nextOutgoingIterator() const {
    assert(m_cb != nullptr);
    return m_cb->nextIterator();
  }

  NodeHandle sourceNode() const {
    assert(m_cb != nullptr);
    return m_cb->sourceNode();
  }

  const LinkedGraph::Edge *ptr() const {
    return m_cb != nullptr ? m_cb->iterator().operator->() : nullptr;
  }

  const E &value() const { return ptr()->value(); }

private:
  CB *m_cb;
};

template <typename V, typename E, typename W = memory::NullWeight,
          typename Allocator = memory::mallocator>
class LinkedGraphMatch {
public:
  using LinkedGraph = memory::LinkedGraph<V, E, W, Allocator>;
  using NodeHandle = LinkedGraph::NodeHandle;
  using EMatch = EdgeMatch<V, E, W, Allocator>;

  LinkedGraphMatch(std::size_t nodeCount, std::size_t edgeCount)
      : m_nodeMatches(nodeCount), m_edgeMatches(edgeCount) {}

  NodeHandle operator[](const NodePatternHandle<V, E, W> &pattern) const {
    return m_nodeMatches[pattern->getId()];
  }
  EMatch operator[](const EdgePatternHandle<V, E, W> &pattern) const {
    return m_edgeMatches[pattern->getId()];
  }

  void registerMatch(const NodePatternHandle<V, E, W> &pattern,
                     const NodeHandle &node) {
    assert(pattern->getId() < m_nodeMatches.size());
    m_nodeMatches[pattern->getId()] = node;
  }

  void registerMatch(const EdgePatternHandle<V, E, W> &pattern, EMatch match) {
    assert(pattern->getId() < m_edgeMatches.size());
    m_edgeMatches[pattern->getId()] = match;
  }

  [[nodiscard]] bool
  mergeMatches(const LinkedGraphMatch<V, E, W, Allocator> &match) {
    assert(match.m_nodeMatches.size() == m_nodeMatches.size());
    assert(match.m_edgeMatches.size() == m_edgeMatches.size());
    for (std::size_t i = 0; i < match.m_nodeMatches.size(); ++i) {
      if (match.m_nodeMatches[i] == nullptr || m_nodeMatches[i] == nullptr) {
        continue;
      }
      if (match.m_nodeMatches[i] != m_nodeMatches[i]) {
        return false; // <- collision!
      }
    }
    for (std::size_t i = 0; i < match.m_edgeMatches.size(); ++i) {
      if (match.m_edgeMatches[i].ptr() == nullptr ||
          m_edgeMatches[i].ptr() == nullptr) {
        continue;
      }
      if (match.m_edgeMatches[i].ptr() != m_edgeMatches[i].ptr()) {
        return false; // <- collision!
      }
    }
    for (std::size_t i = 0; i < match.m_nodeMatches.size(); ++i) {
      if (match.m_nodeMatches[i] != nullptr) {
        m_nodeMatches[i] = match.m_nodeMatches[i];
      }
    }
    for (std::size_t i = 0; i < match.m_edgeMatches.size(); ++i) {
      if (match.m_edgeMatches[i].ptr() != nullptr) {
        m_edgeMatches[i] = match.m_edgeMatches[i];
      }
    }
    return true;
  }

private:
  std::vector<NodeHandle> m_nodeMatches;
  std::vector<EMatch> m_edgeMatches;
};

} // namespace denox::algorithm
