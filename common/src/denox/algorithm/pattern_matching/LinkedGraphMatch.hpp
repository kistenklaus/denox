#pragma once

#include "denox/algorithm/pattern_matching/EdgePattern.fwd.hpp"
#include "denox/algorithm/pattern_matching/NodePattern.fwd.hpp"
#include "denox/memory/allocator/mallocator.hpp"
#include "denox/memory/hypergraph/LinkedGraph.hpp"
#include "denox/memory/hypergraph/NullWeight.hpp"
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace denox::algorithm {

namespace pattern_matching::details {
enum class MatchDirection { Outgoing, Incoming };

template <typename V, typename E, typename W = memory::NullWeight,
          typename Allocator = memory::mallocator>
struct EdgeMatchControl {
  using LinkedGraph = memory::LinkedGraph<V, E, W, Allocator>;
  using NodeHandle = LinkedGraph::NodeHandle;
  using EdgeIt = typename LinkedGraph::EdgeIt;
  using Context = std::vector<EdgeMatchControl *>;

  EdgeMatchControl(NodeHandle node, EdgeIt it,
                   MatchDirection direction = MatchDirection::Outgoing,
                   Context *context = nullptr)
      : m_node(std::move(node)), m_iterator(it), m_direction(direction),
        m_context(context) {
    m_edge = it.operator->();
    m_destination = m_edge->dst();
    if (direction == MatchDirection::Outgoing) {
      m_source = m_node;
      m_outgoing = it;
    } else if (!m_edge->srcs().empty()) {
      m_source = NodeHandle(*m_edge->srcs().begin());
      m_outgoing = m_source->outgoing().begin();
      while (m_outgoing != m_source->outgoing().end() &&
             m_outgoing.operator->() != m_edge)
        ++m_outgoing;
      assert(m_outgoing != m_source->outgoing().end());
    }
    if (m_context)
      m_context->push_back(this);
  }

  EdgeMatchControl(const EdgeMatchControl &) = delete;
  EdgeMatchControl &operator=(const EdgeMatchControl &) = delete;
  ~EdgeMatchControl() {
    if (m_context)
      m_context->erase(std::find(m_context->begin(), m_context->end(), this));
  }

  void erase() {
    if (m_dirty)
      throw std::logic_error("EdgeMatch: edge already erased");
    // Capture every alias's continuation in its own adjacency list before
    // deleting the physical edge. Destination pinning prevents a cascade here.
    auto invalidate = [&](EdgeMatchControl *control) {
      if (control->m_dirty) {
        const auto end = control->m_direction == MatchDirection::Outgoing
                             ? control->m_node->outgoing().end()
                             : control->m_node->incoming().end();
        if (control->m_next != end && control->m_next.operator->() == m_edge)
          ++control->m_next;
        if (control->m_source != nullptr &&
            control->m_nextOutgoing != control->m_source->outgoing().end() &&
            control->m_nextOutgoing.operator->() == m_edge)
          ++control->m_nextOutgoing;
      }
      if (!control->m_dirty && control->m_edge == m_edge) {
        control->m_next = control->m_iterator;
        ++control->m_next;
        if (control->m_source != nullptr) {
          control->m_nextOutgoing = control->m_outgoing;
          ++control->m_nextOutgoing;
        }
        control->m_dirty = true;
      }
    };
    if (m_context) {
      for (auto *control : *m_context)
        invalidate(control);
    } else {
      invalidate(this);
    }
    if (m_direction == MatchDirection::Outgoing)
      m_next = m_node->outgoing().erase(m_iterator);
    else
      m_next = m_node->incoming().erase(m_iterator);
  }

  EdgeIt iterator() const {
    requireSource();
    if (m_dirty)
      throw std::logic_error("EdgeMatch: edge already erased");
    return m_outgoing;
  }
  NodeHandle sourceNode() const { return m_source; }
  const typename LinkedGraph::Edge *ptr() const {
    return m_dirty ? nullptr : m_edge;
  }
  EdgeIt nextOutgoingIterator() const {
    requireSource();
    if (m_dirty)
      return m_nextOutgoing;
    auto next = m_outgoing;
    return ++next;
  }

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
  void requireSource() const {
    if (m_source == nullptr)
      throw std::logic_error(
          "EdgeMatch: source-free edge has no outgoing cursor");
  }
  bool m_dirty = false;
  NodeHandle m_node;
  NodeHandle m_source;
  NodeHandle m_destination;
  typename LinkedGraph::Edge *m_edge;
  LinkedGraph::EdgeIt m_iterator;
  LinkedGraph::EdgeIt m_next;
  EdgeIt m_outgoing;
  EdgeIt m_nextOutgoing;
  MatchDirection m_direction;
  Context *m_context;
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
    return m_cb->nextOutgoingIterator();
  }

  NodeHandle sourceNode() const {
    assert(m_cb != nullptr);
    return m_cb->sourceNode();
  }

  const LinkedGraph::Edge *ptr() const {
    return m_cb != nullptr ? m_cb->ptr() : nullptr;
  }
  bool erased() const { return m_cb && m_cb->dirty(); }

  const E &value() const {
    const auto *edge = ptr();
    if (!edge)
      throw std::logic_error("EdgeMatch: no live edge");
    return edge->value();
  }

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

  bool valid() const {
    return std::none_of(m_edgeMatches.begin(), m_edgeMatches.end(),
                        [](const auto &edge) { return edge.erased(); });
  }

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
