#pragma once

#include "algorithm/lcm.hpp"
#include "memory/allocator/allocator_reference.hpp"
#include "memory/allocator/mallocator.hpp"
#include "memory/allocator/monotone_pool_allocator.hpp"
#include "memory/container/span.hpp"
#include "memory/hypergraph/NullWeight.hpp"
#include <algorithm>
#include <cassert>
#include <concepts>
#include <iterator>
#include <memory>
#include <spdlog/pattern_formatter-inl.h>
#include <type_traits>

namespace denox::memory {

template <typename V, typename E, typename W = NullWeight,
          typename Allocator = mallocator>
class LinkedGraph {
public:
  class Node;
  class NodeHandle;
  class Edge;

private:
  using ref_count = std::uint32_t;
  struct OutgoingNodeList;
  struct EdgeSrc;
  struct EdgeList;
  struct ControlBlock;

  struct OutgoingNodeList {
    OutgoingNodeList *next;
    EdgeList *outgoingEntry;
    Node *node;

    OutgoingNodeList() noexcept = default;
    OutgoingNodeList(OutgoingNodeList *next, EdgeList *outgoingEntry,
                     Node *node) noexcept
        : next(next), outgoingEntry(outgoingEntry), node(node) {}
  };

  struct EdgeList {
    EdgeList *next;
    EdgeList *prev;
    Edge *info;

    EdgeList() noexcept = default;
    EdgeList(EdgeList *next, EdgeList *prev, Edge *info) noexcept
        : next(next), prev(prev), info(info) {}
  };

  struct ControlBlock {
    static_assert(sizeof(OutgoingNodeList) == sizeof(EdgeList));
    static constexpr std::size_t CommonBlockSize = algorithm::lcm(
        sizeof(EdgeList), algorithm::lcm(sizeof(Node), sizeof(Edge)));

    static constexpr std::size_t MaxBlockSize = 1 << 14;
    static constexpr std::size_t EffectiveBlockSize =
        std::min(CommonBlockSize,
                 std::max(std::max(sizeof(Node),
                                   std::max(sizeof(Edge), sizeof(EdgeList))),
                          MaxBlockSize));

    static constexpr std::size_t CommonAlign =
        std::max(alignof(EdgeList), std::max(alignof(Node), alignof(Edge)));

    using BlockAlloc =
        monotonic_pool_allocator<EffectiveBlockSize, CommonAlign, Allocator>;
    BlockAlloc blockPool;

    static constexpr std::size_t LinkedPoolBlockCapacity =
        EffectiveBlockSize / sizeof(EdgeList);
    monotonic_pool_allocator<sizeof(EdgeList), alignof(EdgeList),
                             allocator_ref<BlockAlloc>, std::ratio<1, 1>>
        linkedPool;

    static constexpr std::size_t NodePoolBlockCapacity =
        EffectiveBlockSize / sizeof(Node);
    monotonic_pool_allocator<sizeof(Node), alignof(Node),
                             allocator_ref<BlockAlloc>, std::ratio<1, 1>>
        nodePool;

    static constexpr std::size_t EdgePoolBlockCapacity =
        EffectiveBlockSize / sizeof(Edge);
    monotonic_pool_allocator<sizeof(Edge), alignof(Edge),
                             allocator_ref<BlockAlloc>, std::ratio<1, 1>>
        edgePool;

    ControlBlock(const Allocator& upstream = {})
        : blockPool(0, upstream),
          linkedPool(LinkedPoolBlockCapacity, allocator_ref(&blockPool)),
          nodePool(NodePoolBlockCapacity, allocator_ref(&blockPool)),
          edgePool(EdgePoolBlockCapacity, allocator_ref(&blockPool)) {
      constexpr auto x = sizeof(EdgeList);
    }

    Node *allocNode() {
      return static_cast<Node *>(
          nodePool.allocate(sizeof(Node), alignof(Node)));
    }
    void destroyNode(Node *nodeBlock) noexcept {
      assert(nodeBlock != nullptr);
      assert(nodeBlock->m_external_count == 0);
      assert(nodeBlock->m_live_parent_count == 0);
      assert(nodeBlock->m_incoming == nullptr);
      if constexpr (!std::is_trivially_destructible_v<Node>) {
        nodeBlock->~Node();
      }
      nodePool.deallocate(nodeBlock);
    }

    Edge *allocEdgeInfo() noexcept {
      return static_cast<Edge *>(
          edgePool.allocate(sizeof(Edge), alignof(Edge)));
    }

    void destroyEdgeInfo(Edge *ptr) noexcept {
      assert(ptr != nullptr);
      if constexpr (!std::is_trivially_destructible_v<Edge>) {
        ptr->~Edge();
      }
      edgePool.deallocate(ptr);
    }

    EdgeList *allocEdgeList() noexcept {
      return static_cast<EdgeList *>(
          linkedPool.allocate(sizeof(EdgeList), alignof(EdgeList)));
    }

    void destroyEdgeList(EdgeList *ptr) noexcept {
      assert(ptr != nullptr);
      static_assert(std::is_trivially_destructible_v<EdgeList>);
      static_assert(sizeof(OutgoingNodeList) == sizeof(EdgeList));
      static_assert(alignof(OutgoingNodeList) == alignof(EdgeList));
      linkedPool.deallocate(ptr);
    }

    OutgoingNodeList *allocOutgoingNodeList() noexcept {
      return static_cast<OutgoingNodeList *>(linkedPool.allocate(
          sizeof(OutgoingNodeList), alignof(OutgoingNodeList)));
    }

    void destroyOutgoingNodeList(OutgoingNodeList *ptr) noexcept {
      assert(ptr != nullptr);
      static_assert(std::is_trivially_destructible_v<OutgoingNodeList>);
      static_assert(sizeof(OutgoingNodeList) == sizeof(EdgeList));
      static_assert(alignof(OutgoingNodeList) == alignof(EdgeList));
      linkedPool.deallocate(ptr);
    }
  };

public:
  // ======================= PUBLIC-INTERFACE ===================

  class Edge {
  public:
    friend LinkedGraph;

    Edge(const Edge &) = delete;
    Edge &operator=(const Edge &) = delete;
    Edge(Edge &&) = delete;
    Edge &operator=(Edge &&) = delete;

    const E &operator*() const noexcept { return m_payload; }

    E &operator*() noexcept { return m_payload; }

    const E *operator->() const noexcept { return &m_payload; }

    E *operator->() noexcept { return &m_payload; }

    E &value() { return m_payload; }

    const E &value() const { return m_payload; }

    const Node &dst() const {
      assert(m_dst != nullptr);
      return *m_dst;
    }
    NodeHandle dst() {
      assert(m_dst != nullptr);
      assert((m_dst->m_external_count + m_dst->m_live_parent_count) > 0ull);
      return NodeHandle(*m_dst);
    }

    class SrcList;
    class SrcIt {
    public:
      friend SrcList;
      using iterator_category = std::forward_iterator_tag;
      using iterator_concept = std::forward_iterator_tag;
      using difference_type = std::ptrdiff_t;
      using value_type = Node;
      using reference = Node &;
      using pointer = Node *;

      SrcIt() : m_src(nullptr) {}

      reference operator*() const noexcept {
        assert(m_src != nullptr);
        assert(m_src->node != nullptr);
        return *m_src->node;
      }

      pointer operator->() const noexcept {
        assert(m_src != nullptr);
        assert(m_src->node != nullptr);
        return m_src->node;
      }

      SrcIt &operator++() {
        assert(m_src != nullptr);
        m_src = m_src->next;
        return *this;
      }

      SrcIt operator++(int) {
        SrcIt it = *this;
        ++(*this);
        return it;
      }

      friend bool operator==(const SrcIt &lhs, const SrcIt &rhs) {
        return lhs.m_src == rhs.m_src;
      }

      friend bool operator!=(const SrcIt &lhs, const SrcIt &rhs) {
        return lhs.m_src != rhs.m_src;
      }

    private:
      explicit SrcIt(const OutgoingNodeList *src) : m_src(src) {}
      const OutgoingNodeList *m_src;
    };

    class SrcList {
    public:
      friend Edge;
      [[nodiscard]] SrcIt begin() const noexcept {
        return SrcIt(m_edge->m_srcs);
      }

      [[nodiscard]] SrcIt end() const noexcept { return SrcIt(); }

      [[nodiscard]] std::size_t size() const noexcept {
        std::size_t size = 0;
        auto it = begin();
        auto e = end();
        while (it++ != e) {
          ++size;
        }
        return size;
      }

    private:
      SrcList(const Edge *edge) : m_edge(edge) {}
      const Edge *m_edge;
    };

    [[nodiscard]] SrcList srcs() const { return SrcList(this); }

  private:
    template <typename... Args>
    explicit Edge(OutgoingNodeList *srcs, EdgeList *incomingEntry, Node *dst,
                  W &&weight, Args &&...args) noexcept
      requires std::constructible_from<E, Args...>
        : m_payload(std::forward<Args>(args)...), m_weight(std::move(weight)),
          m_srcs(srcs), m_incomingEntry(incomingEntry), m_dst(dst) {
      assert(srcs != nullptr);
      assert(incomingEntry != nullptr);
      assert(dst != nullptr);
    }

    [[no_unique_address]] E m_payload;
    [[no_unique_address]] W m_weight;
    OutgoingNodeList *m_srcs;
    EdgeList *m_incomingEntry;
    Node *m_dst;
  };

  class OutgoingList;
  class IncomingList;

  class Node {
  public:
    friend LinkedGraph;

    Node(const Node &) = delete;
    Node &operator=(const Node &) = delete;
    Node(Node &&) = delete;
    Node &operator=(Node &&) = delete;

    NodeHandle handle() { return NodeHandle(*this); }

    const V &operator*() const noexcept { return m_payload; }
    V &operator*() noexcept { return m_payload; }
    const V *operator->() const noexcept { return &m_payload; }
    V *operator->() noexcept { return &m_payload; }
    const V &value() const noexcept { return m_payload; }
    V &value() noexcept { return m_payload; }

    [[nodiscard]] IncomingList incoming() noexcept {
      return IncomingList(NodeHandle(*this));
    }

    [[nodiscard]] OutgoingList outgoing() noexcept {
      return OutgoingList(NodeHandle(*this));
    }

  private:
    ref_count m_external_count;
    ref_count m_live_parent_count;
    V m_payload;
    ControlBlock *m_controlBlock;
    EdgeList *m_outgoing;
    EdgeList *m_incoming; // storage.

    template <typename... Args>
    Node(ControlBlock *controlBlock, Args &&...args) noexcept
        : m_external_count(1), m_live_parent_count(0),
          m_payload(std::forward<Args>(args)...), m_controlBlock(controlBlock),
          m_outgoing(nullptr), m_incoming(nullptr) {}

    void decExternalCount() noexcept {
      assert(m_external_count > 0);
      m_external_count -= 1;
      cascadeLifeness();
    }

    void decLiveParentCount() noexcept {
      assert(m_live_parent_count > 0);
      m_live_parent_count -= 1;
      cascadeLifeness();
    }

    void cascadeLifeness() noexcept {
      if (m_external_count > 0 || m_live_parent_count > 0) {
        return;
      }
      OutgoingNodeList queue{nullptr, nullptr, nullptr};
      OutgoingNodeList *tail = &queue;
      ControlBlock *cb = m_controlBlock;

      enqueueDeadChildren(this, tail);
      cb->destroyNode(this);
      // DO NOT TOUCH MEMBERS AFTER THIS POINT!

      OutgoingNodeList *head = queue.next;
      while (head != nullptr) {
        enqueueDeadChildren(head->node, tail);
        OutgoingNodeList *next = head->next;
        cb->destroyNode(head->node);
        cb->destroyOutgoingNodeList(head);
        head = next;
      }
    }

    static void enqueueDeadChildren(Node *node,
                                    OutgoingNodeList *&queue) noexcept {
      assert(queue != nullptr);
      // NOTE: Dead nodes cannot have incoming edges!!
      assert(node->m_incoming == nullptr);

      // 1. Remove all outgoing edges, while accumulating a list of children.
      EdgeList *curr = node->m_outgoing;
      if (curr != nullptr) {
        const void *outgoingHead = static_cast<const void *>(node->m_outgoing);
        do {
          Edge *edgeInfo = curr->info;
          // Remove all outgoing edges from srcs nodes, while skipping self.
          OutgoingNodeList *srcs = edgeInfo->m_srcs;
          while (srcs != nullptr) {
            if (srcs->node != node) {
              EdgeList *outgoingEntry = srcs->outgoingEntry;
              assert(outgoingEntry != nullptr);
              if (outgoingEntry->prev == outgoingEntry) {
                assert(outgoingEntry->next == outgoingEntry);
                // node.outgoing list has size 1! (cyclic list)
                // => collapse to nullptr.
                srcs->node->m_outgoing = nullptr;
              } else {
                outgoingEntry->prev->next = outgoingEntry->next;
                outgoingEntry->next->prev = outgoingEntry->prev;
                if (outgoingEntry == srcs->node->m_outgoing) {
                  // remount head.
                  srcs->node->m_outgoing = outgoingEntry->next;
                }
              }
              node->m_controlBlock->destroyEdgeList(outgoingEntry);
            }
            OutgoingNodeList *srcNext = srcs->next;
            node->m_controlBlock->destroyOutgoingNodeList(srcs);
            srcs = srcNext;
          }
          // Remove all incoming edge from dst node
          Node *dst = edgeInfo->m_dst;
          EdgeList *dstIncoming = edgeInfo->m_incomingEntry;
          assert(dstIncoming != nullptr);
          if (dstIncoming->prev == dstIncoming) {
            assert(dstIncoming->next == dstIncoming);
            // node.incoming has list size 1! (cyclic list)
            // => collapse to nullptr
            dst->m_incoming = nullptr;
          } else {
            dstIncoming->prev->next = dstIncoming->next;
            dstIncoming->next->prev = dstIncoming->prev;
            if (dst->m_incoming == dstIncoming) {
              // remount head.
              dst->m_incoming = dstIncoming->next;
            }
          }
          node->m_controlBlock->destroyEdgeInfo(dstIncoming->info);
          node->m_controlBlock->destroyEdgeList(dstIncoming);

          EdgeList *next = curr->next;
          assert(dst->m_live_parent_count > 0);
          dst->m_live_parent_count -= 1;
          if (dst->m_external_count == 0 && dst->m_live_parent_count == 0) {
            curr->~EdgeList();
            // NOTE: Super evil memory reuse trick
            OutgoingNodeList *temp = reinterpret_cast<OutgoingNodeList *>(curr);
            new (temp) OutgoingNodeList(nullptr, nullptr, dst);
            // push to linked list!
            queue->next = temp;
            queue = temp;
          } else {
            node->m_controlBlock->destroyEdgeList(curr);
          }
          curr = next;
        } while (static_cast<const void *>(curr) != outgoingHead);
      }
      node->m_outgoing = nullptr;
    }
  };

  class EdgeIt {
  public:
    friend NodeHandle;
    friend Node;
    friend OutgoingList;
    friend IncomingList;

    using iterator_category = std::input_iterator_tag;
    using iterator_concept = std::input_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = Edge;
    using reference = Edge &;
    using pointer = Edge *;

    EdgeIt() noexcept : m_head(nullptr), m_curr(nullptr) {}

    reference operator*() const noexcept {
      assert(m_curr != nullptr);
      return *m_curr->info;
    }

    pointer operator->() const noexcept {
      assert(m_curr != nullptr);
      return m_curr->info;
    }

    EdgeIt &operator++() noexcept {
      assert(m_head != nullptr);
      if (m_curr != nullptr) {
        if (m_curr->next == *m_head) {
          m_curr = nullptr;
        } else {
          m_curr = m_curr->next;
        }
      } else {
        m_curr = *m_head;
      }
      return *this;
    }

    EdgeIt operator++(int) noexcept {
      EdgeIt tmp = *this;
      ++(*this);
      return tmp;
    }

    EdgeIt &operator--() noexcept {
      assert(m_head != nullptr);
      if (m_curr == nullptr) {
        if (*m_head != nullptr) {
          m_curr = (*m_head)->prev;
        } // else stay end!
      } else {
        m_curr = m_curr->prev;
      }
      return *this;
    }

    EdgeIt operator--(int) noexcept {
      EdgeIt tmp = *this;
      --(*this);
      return tmp;
    }

    friend bool operator==(const EdgeIt &lhs, const EdgeIt &rhs) noexcept {
      return lhs.m_head == rhs.m_head && lhs.m_curr == rhs.m_curr;
    }

    friend bool operator!=(const EdgeIt &lhs, const EdgeIt &rhs) noexcept {
      return !(lhs == rhs);
    }

  private:
    EdgeIt(EdgeList *const *head, EdgeList *curr) noexcept
        : m_head(head), m_curr(curr) {}
    EdgeList *const *m_head;
    EdgeList *m_curr;
  };

  class OutgoingList {
  public:
    friend NodeHandle;
    friend Node;

    EdgeIt erase(EdgeIt pos) noexcept {
      assert(m_node.m_controlBlock != nullptr);
      assert(pos.m_head == &m_node.m_controlBlock->m_outgoing);
      assert(pos.m_curr != nullptr); // <- cannot erase from an empty list!
      // 1. Mark as a sentinel!
      // NOTE: outgoing does not have ownership,
      // incoming holds ownership of EdgeInfo.
      auto cb = m_node.m_controlBlock->m_controlBlock;
      assert(cb != nullptr);
      EdgeList *entry = pos.m_curr;
      {
        Edge *info = entry->info;
        entry->info = nullptr;
        LinkedGraph::removeEdge_Internal(cb, info);
      }
      // Still a sentinal, nobody touched it.
      assert(entry->info == nullptr);
      if (entry->prev == entry) {
        assert(entry->next == entry);
        m_node.m_controlBlock->m_outgoing = nullptr;
        cb->destroyEdgeList(entry);
        return end();
      } else {
        ++pos;
        EdgeList *next = entry->next;
        EdgeList *prev = entry->prev;
        prev->next = next;
        next->prev = prev;
        if (entry == m_node.m_controlBlock->m_outgoing) {
          m_node.m_controlBlock->m_outgoing = next;
        }
        cb->destroyEdgeList(entry);
        return pos;
      }
    }

    template <typename... Args>
      requires std::constructible_from<E, Args...>
    EdgeIt insert_after_with_dynamic_srcs(
        EdgeIt pos,
        memory::span<const NodeHandle *> additionalSources, // <- can be empty.
        const NodeHandle &dst, W weight, Args &&...args) noexcept {
      assert(m_node.m_controlBlock != nullptr);
      assert(pos.m_head == &m_node.m_controlBlock->m_outgoing);
      auto cb = m_node.m_controlBlock->m_controlBlock;
      const NodeHandle &src0 = m_node;
      const auto src0Node = src0.m_controlBlock;

      assert(dst.m_controlBlock != nullptr);
      Node *dstNode = dst.m_controlBlock;
      assert(dstNode->m_controlBlock == cb);
      assert(dstNode != src0Node);

      Edge *edgeInfo = cb->allocEdgeInfo();

      OutgoingNodeList *srcsList = nullptr;
      for (int s = static_cast<int>(additionalSources.size()) - 1; s >= 0;
           --s) {
        const NodeHandle *ptr = additionalSources[static_cast<std::size_t>(s)];
        assert(ptr && ptr->m_controlBlock &&
               ptr->m_controlBlock->m_controlBlock == cb);
        assert(ptr->m_controlBlock != dstNode);
        const NodeHandle &src = *ptr;
        EdgeList *outgoingEntry = cb->allocEdgeList();
        new (outgoingEntry) EdgeList(nullptr, nullptr, edgeInfo);
        auto &srcNode = src.m_controlBlock;
        assert(srcNode->m_controlBlock == cb);
        if (srcNode->m_outgoing == nullptr) {
          outgoingEntry->prev = outgoingEntry;
          outgoingEntry->next = outgoingEntry;
          srcNode->m_outgoing = outgoingEntry;
        } else {
          outgoingEntry->next = srcNode->m_outgoing->next;
          outgoingEntry->prev = srcNode->m_outgoing;
          srcNode->m_outgoing->next->prev = outgoingEntry;
          srcNode->m_outgoing->next = outgoingEntry;
        }
        OutgoingNodeList *srcsEntry = cb->allocOutgoingNodeList();
        new (srcsEntry) OutgoingNodeList(srcsList, outgoingEntry, srcNode);
        srcsList = srcsEntry;
      }
      EdgeList *outgoingEntry = cb->allocEdgeList();
      new (outgoingEntry) EdgeList(nullptr, nullptr, edgeInfo);
      if (src0Node->m_outgoing == nullptr) {
        assert(pos == end()); // <- empty list.
        outgoingEntry->prev = outgoingEntry;
        outgoingEntry->next = outgoingEntry;
        src0Node->m_outgoing = outgoingEntry;
      } else {
        assert(pos.m_curr != nullptr);
        outgoingEntry->next = pos.m_curr->next;
        outgoingEntry->prev = pos.m_curr;
        pos.m_curr->next->prev = outgoingEntry;
        pos.m_curr->next = outgoingEntry;
      }
      OutgoingNodeList *srcs = cb->allocOutgoingNodeList();
      new (srcs) OutgoingNodeList(srcsList, outgoingEntry, src0Node);

      EdgeList *incomingEntry = cb->allocEdgeList();
      new (incomingEntry) EdgeList(nullptr, nullptr, edgeInfo);
      if (dstNode->m_incoming == nullptr) {
        incomingEntry->next = incomingEntry;
        incomingEntry->prev = incomingEntry;
        dstNode->m_incoming = incomingEntry;
      } else {
        incomingEntry->next = dstNode->m_incoming->next;
        incomingEntry->prev = dstNode->m_incoming;
        dstNode->m_incoming->next->prev = incomingEntry;
        dstNode->m_incoming->next = incomingEntry;
      }

      new (edgeInfo) Edge(srcs, incomingEntry, dst.m_controlBlock,
                          std::move(weight), std::forward<Args>(args)...);
      dstNode->m_live_parent_count += 1;
      return ++pos;
    }

    template <typename... Args>
      requires std::default_initializable<W> &&
               std::constructible_from<E, Args...>
    EdgeIt insert_after(EdgeIt pos, const NodeHandle &dst,
                        Args &&...args) noexcept {
      return insert_after_with_dynamic_srcs(
          pos, memory::span<const NodeHandle *>(), dst, W{},
          std::forward<Args>(args)...);
    }

    template <typename... Args>
      requires(!std::default_initializable<W> &&
               std::constructible_from<E, Args...>)
    EdgeIt insert_after(EdgeIt pos, const NodeHandle &dst, const W &weight,
                        Args &&...args) noexcept {
      return insert_after_with_dynamic_srcs(
          pos, memory::span<const NodeHandle *>(), dst, weight,
          std::forward<Args>(args)...);
    }

    template <typename... Args>
      requires std::default_initializable<W> &&
               std::constructible_from<E, Args...>
    EdgeIt insert_after(EdgeIt pos, const NodeHandle &src1,
                        const NodeHandle &dst, Args &&...args) noexcept {
      const NodeHandle *ptr = &src1;
      return insert_after_with_dynamic_srcs(
          pos, memory::span<const NodeHandle *>{&ptr, 1}, dst, W{},
          std::forward<Args>(args)...);
    }

    /// Inserts anywhere
    template <typename... Args>
      requires std::default_initializable<W> &&
               std::constructible_from<E, Args...>
    EdgeIt insert(const NodeHandle &dst, Args &&...args) noexcept {
      return insert_after_with_dynamic_srcs(
          begin(), memory::span<const NodeHandle *>(), dst, W{},
          std::forward<Args>(args)...);
    }

    template <typename... Args>
      requires(!std::default_initializable<W> &&
               std::constructible_from<E, Args...>)
    EdgeIt insert(const NodeHandle &dst, const W &weight,
                  Args &&...args) noexcept {
      return insert_after_with_dynamic_srcs(
          begin(), memory::span<const NodeHandle *>(), dst, weight,
          std::forward<Args>(args)...);
    }

    /// Inserts anywhere
    template <typename... Args>
      requires std::default_initializable<W> &&
               std::constructible_from<E, Args...>
    EdgeIt insert(const NodeHandle &src1, const NodeHandle &dst,
                  Args &&...args) noexcept {
      const NodeHandle *ptr = &src1;
      return insert_after_with_dynamic_srcs(
          begin(), memory::span<const NodeHandle *>(&ptr, 1), dst, W{},
          std::forward<Args>(args)...);
    }
    template <typename... Args>
      requires(!std::default_initializable<W> &&
               std::constructible_from<E, Args...>)
    EdgeIt insert(const NodeHandle &src1, const NodeHandle &dst,
                  const W &weight, Args &&...args) noexcept {
      const NodeHandle *ptr = &src1;
      return insert_after_with_dynamic_srcs(
          begin(), memory::span<const NodeHandle *>(&ptr, 1), dst, weight,
          std::forward<Args>(args)...);
    }

    [[nodiscard]] EdgeIt begin() noexcept {
      return EdgeIt(&m_node.m_controlBlock->m_outgoing,
                    m_node.m_controlBlock->m_outgoing);
    }

    [[nodiscard]] EdgeIt end() noexcept {
      return EdgeIt(&m_node.m_controlBlock->m_outgoing, nullptr);
    }

    [[nodiscard]] std::size_t size() noexcept {
      std::size_t size = 0;
      auto it = begin();
      auto e = end();
      while (it++ != e) {
        ++size;
      }
      return size;
    }

  private:
    explicit OutgoingList(NodeHandle node) : m_node(std::move(node)) {}
    NodeHandle m_node;
  };

  class IncomingList {
  public:
    friend NodeHandle;
    friend Node;

    EdgeIt erase(EdgeIt pos) noexcept {
      // NOTE: Because we hold a external_count of m_node, we are guaranteed
      // that this operation will never cascade!
      // Beccause of this just going ++next is actually valid, otherwise
      // next might point to a edge which was deleted in a cascade!
      assert(pos.m_head == &m_node.m_controlBlock->m_incoming);
      EdgeIt next = pos;
      ++next;
      LinkedGraph::removeEdge_Internal(m_node.m_controlBlock->m_controlBlock,
                                       pos.m_curr->info);
      return next;
    }

    template <typename... Args>
    EdgeIt insert_after_with_dynamic_srcs(EdgeIt pos,
                                          memory::span<const NodeHandle *> srcs,
                                          W weight, Args &&...args) noexcept {
      const NodeHandle &dst = m_node;

      Node *dstNode = dst.m_controlBlock;
      ControlBlock *cb = dstNode->m_controlBlock;
      assert(cb != nullptr);
      assert(pos.m_head == &dstNode->m_incoming);
      assert(dstNode->m_controlBlock == cb);
      assert(!srcs.empty());

      Edge *edgeInfo = cb->allocEdgeInfo();

      OutgoingNodeList *srcsList = nullptr;
      for (int s = static_cast<int>(srcs.size()) - 1; s >= 0; --s) {
        const NodeHandle *ptr = srcs[static_cast<std::size_t>(s)];
        assert(ptr && ptr->m_controlBlock &&
               ptr->m_controlBlock->m_controlBlock == cb);
        assert(ptr->m_controlBlock != dstNode);
        const NodeHandle &src = *srcs[static_cast<std::size_t>(s)];
        EdgeList *outgoingEntry = cb->allocEdgeList();
        new (outgoingEntry) EdgeList(nullptr, nullptr, edgeInfo);
        auto &srcNode = src.m_controlBlock;
        assert(srcNode->m_controlBlock == cb);
        if (srcNode->m_outgoing == nullptr) {
          outgoingEntry->prev = outgoingEntry;
          outgoingEntry->next = outgoingEntry;
          srcNode->m_outgoing = outgoingEntry;
        } else {
          outgoingEntry->next = srcNode->m_outgoing->next;
          outgoingEntry->prev = srcNode->m_outgoing;
          srcNode->m_outgoing->next->prev = outgoingEntry;
          srcNode->m_outgoing->next = outgoingEntry;
        }
        OutgoingNodeList *srcsEntry = cb->allocOutgoingNodeList();
        new (srcsEntry) OutgoingNodeList(srcsList, outgoingEntry, srcNode);
        srcsList = srcsEntry;
      }

      EdgeList *incomingEntry = cb->allocEdgeList();
      new (incomingEntry) EdgeList(nullptr, nullptr, edgeInfo);
      if (dstNode->m_incoming == nullptr) {
        assert(pos == end()); // must be empty!
        incomingEntry->next = incomingEntry;
        incomingEntry->prev = incomingEntry;
        dstNode->m_incoming = incomingEntry;
        // NOTE: ++pos, now points to incomingEntry!,
        // recovers m_curr == nullptr, by reloading the head!
      } else {
        assert(pos.m_curr != nullptr);
        incomingEntry->next = pos.m_curr->next;
        incomingEntry->prev = pos.m_curr;
        pos.m_curr->next->prev = incomingEntry;
        pos.m_curr->next = incomingEntry;
      }

      new (edgeInfo) Edge(srcsList, incomingEntry, dstNode, std::move(weight),
                          std::forward<Args>(args)...);
      dstNode->m_live_parent_count += 1;
      return ++pos;
    }

    template <typename... Args>
      requires std::default_initializable<W> &&
               std::constructible_from<E, Args...>
    EdgeIt insert_after(EdgeIt pos, const NodeHandle &src,
                        Args &&...args) noexcept {
      const NodeHandle *ptr = &src;
      return insert_after_with_dynamic_srcs(pos, memory::span{&ptr, 1}, W{},
                                            std::forward<Args>(args)...);
    }
    template <typename... Args>
      requires(!std::default_initializable<W> &&
               std::constructible_from<E, Args...>)
    EdgeIt insert_after(EdgeIt pos, const NodeHandle &src, const W &weight,
                        Args &&...args) noexcept {
      const NodeHandle *ptr = &src;
      return insert_after_with_dynamic_srcs(pos, memory::span{&ptr, 1}, weight,
                                            std::forward<Args>(args)...);
    }

    template <typename... Args>
      requires std::default_initializable<W> &&
               std::constructible_from<E, Args...>
    EdgeIt insert_after(EdgeIt pos, const NodeHandle &src0,
                        const NodeHandle &src1, Args &&...args) noexcept {
      const NodeHandle *srcs[] = {&src0, &src1};
      return insert_after_with_dynamic_srcs(pos, srcs, W{},
                                            std::forward<Args>(args)...);
    }
    template <typename... Args>
      requires(!std::default_initializable<W> &&
               std::constructible_from<E, Args...>)
    EdgeIt insert_after(EdgeIt pos, const NodeHandle &src0,
                        const NodeHandle &src1, const W &weight,
                        Args &&...args) noexcept {
      const NodeHandle *srcs[] = {&src0, &src1};
      return insert_after_with_dynamic_srcs(pos, srcs, weight,
                                            std::forward<Args>(args)...);
    }

    template <typename... Args>
      requires std::default_initializable<W> &&
               std::constructible_from<E, Args...>
    EdgeIt insert(const NodeHandle &src0, const NodeHandle &src1,
                  Args &&...args) noexcept {
      const NodeHandle *srcs[] = {&src0, &src1};
      return insert_after_with_dynamic_srcs(begin(), srcs, W{},
                                            std::forward<Args>(args)...);
    }
    template <typename... Args>
      requires(!std::default_initializable<W> &&
               std::constructible_from<E, Args...>)
    EdgeIt insert(const NodeHandle &src0, const NodeHandle &src1,
                  const W &weight, Args &&...args) noexcept {
      const NodeHandle *srcs[] = {&src0, &src1};
      return insert_after_with_dynamic_srcs(begin(), srcs, weight,
                                            std::forward<Args>(args)...);
    }

    template <typename... Args>
      requires std::default_initializable<W> &&
               std::constructible_from<E, Args...>
    EdgeIt insert(const NodeHandle &src, Args &&...args) noexcept {
      const NodeHandle *ptr = &src;
      return insert_after_with_dynamic_srcs(begin(), memory::span{&ptr, 1}, W{},
                                            std::forward<Args>(args)...);
    }
    template <typename... Args>
      requires(!std::default_initializable<W> &&
               std::constructible_from<E, Args...>)
    EdgeIt insert(const NodeHandle &src, const W &weight,
                  Args &&...args) noexcept {
      const NodeHandle *ptr = &src;
      return insert_after_with_dynamic_srcs(
          begin(), memory::span{&ptr, 1}, weight, std::forward<Args>(args)...);
    }

    [[nodiscard]] EdgeIt begin() noexcept {
      return EdgeIt(&m_node.m_controlBlock->m_incoming,
                    m_node.m_controlBlock->m_incoming);
    }
    [[nodiscard]] EdgeIt end() noexcept {
      return EdgeIt(&m_node.m_controlBlock->m_incoming, nullptr);
    }

    [[nodiscard]] std::size_t size() noexcept {
      std::size_t size = 0;
      auto it = begin();
      auto e = end();
      while (it++ != e) {
        ++size;
      }
      return size;
    }

  private:
    explicit IncomingList(NodeHandle node) noexcept : m_node(std::move(node)) {}
    NodeHandle m_node; // <- holds external_count!
  };

  class NodeHandle {
  public:
    friend LinkedGraph;
    friend Edge;
    friend Node;
    friend OutgoingList;
    friend IncomingList;

    NodeHandle(Node &o) noexcept : m_controlBlock(&o) {
      m_controlBlock->m_external_count += 1;
    }

    NodeHandle(const NodeHandle &o) noexcept
        : m_controlBlock(o.m_controlBlock) {
      if (m_controlBlock != nullptr) {
        m_controlBlock->m_external_count += 1;
      }
    }
    NodeHandle &operator=(const NodeHandle &o) noexcept {
      if (this == &o) {
        return *this;
      }
      release();
      m_controlBlock = o.m_controlBlock;
      if (m_controlBlock != nullptr) {
        m_controlBlock->m_external_count += 1;
      }
      return *this;
    }

    NodeHandle(NodeHandle &&o) noexcept
        : m_controlBlock(std::exchange(o.m_controlBlock, nullptr)) {}
    NodeHandle &operator=(NodeHandle &&o) noexcept {
      if (this == &o) {
        return *this;
      }
      release();
      m_controlBlock = std::exchange(o.m_controlBlock, nullptr);
      return *this;
    }

    ~NodeHandle() noexcept { release(); }

    const Node &operator*() const noexcept {
      assert(m_controlBlock != nullptr);
      return *m_controlBlock;
    }
    Node &operator*() noexcept {
      assert(m_controlBlock != nullptr);
      return *m_controlBlock;
    }
    const Node *operator->() const noexcept {
      assert(m_controlBlock != nullptr);
      return m_controlBlock;
    }
    Node *operator->() noexcept {
      assert(m_controlBlock != nullptr);
      return m_controlBlock;
    }

    void release() noexcept {
      if (m_controlBlock != nullptr) {
        m_controlBlock->decExternalCount();
        m_controlBlock = nullptr;
      }
    }

  private:
    explicit NodeHandle(Node *cb) noexcept : m_controlBlock(cb) {}
    Node *m_controlBlock;
  };
  friend NodeHandle;

private:
  // ============= STATIC-HELPERS ==================
  static void removeEdge_Internal(ControlBlock *controlBlock,
                                  Edge *edgeInfo) noexcept {
    // 1. Remove outgoing from srcs, while deallocating.
    OutgoingNodeList *srcs = edgeInfo->m_srcs;
    while (srcs != nullptr) {
      EdgeList *outgoingEntry = srcs->outgoingEntry;
      assert(outgoingEntry != nullptr);
      // NOTE: info == nullptr, means this is a node sentinal
      // ignore it removal is handled somewhere else!
      if (outgoingEntry->info != nullptr) {
        if (outgoingEntry->prev == outgoingEntry) {
          assert(outgoingEntry->next == outgoingEntry);
          srcs->node->m_outgoing = nullptr;
        } else {
          outgoingEntry->prev->next = outgoingEntry->next;
          outgoingEntry->next->prev = outgoingEntry->prev;
          if (outgoingEntry == srcs->node->m_outgoing) {
            srcs->node->m_outgoing = outgoingEntry->next;
          }
        }
        controlBlock->destroyEdgeList(outgoingEntry);
      }
      OutgoingNodeList *next = srcs->next;
      controlBlock->destroyOutgoingNodeList(srcs);
      srcs = next;
    }
    // 2. Remove incoming entry and deallocate.
    EdgeList *incomingEntry = edgeInfo->m_incomingEntry;
    if (incomingEntry->prev == incomingEntry) {
      assert(incomingEntry->next == incomingEntry);
      edgeInfo->m_dst->m_incoming = nullptr;
    } else {
      incomingEntry->prev->next = incomingEntry->next;
      incomingEntry->next->prev = incomingEntry->prev;
      if (incomingEntry == edgeInfo->m_dst->m_incoming) {
        edgeInfo->m_dst->m_incoming = incomingEntry->next;
      }
    }
    controlBlock->destroyEdgeList(incomingEntry);
    // 3. Decrement live_parent_count of dst.
    edgeInfo->m_dst->decLiveParentCount();
    controlBlock->destroyEdgeInfo(edgeInfo);
  }

public:
  template <typename... Args>
    requires std::constructible_from<V, Args...>
  NodeHandle createNode(Args &&...args) noexcept {
    Node *nodeBlock = m_controlBlock->allocNode();
    new (nodeBlock) Node(m_controlBlock.get(), std::forward<Args>(args)...);
    return NodeHandle{nodeBlock};
  }

  LinkedGraph(const Allocator& alloc = {})
      : m_controlBlock(std::make_unique<ControlBlock>(alloc)) {}

private:
  std::unique_ptr<ControlBlock> m_controlBlock;
};

} // namespace denox::memory
