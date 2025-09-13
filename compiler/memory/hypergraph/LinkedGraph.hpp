#pragma once

#include "memory/allocator/monotone_pool_allocator.hpp"
#include "memory/hypergraph/NullWeight.hpp"
#include <cassert>
#include <concepts>
#include <memory>
#include <type_traits>
namespace denox::memory {

template <typename V, typename E, typename W = NullWeight> class LinkedGraph {
private:
  using ref_count = std::uint32_t;
  struct OutgoingNodeList;
  struct EdgeSrc;
  struct EdgeList;
  struct ControlBlock;
  struct NodeControlBlock;

  struct EdgeInfo {
    E payload;
    OutgoingNodeList *srcs;
    EdgeList *incomingEntry;
    NodeControlBlock *dst;

    template <typename... Args>
    explicit EdgeInfo(OutgoingNodeList *srcs, EdgeList *incomingEntry,
                      NodeControlBlock *dst, Args &&...args)
        : payload(std::forward<Args>(args)...), srcs(srcs),
          incomingEntry(incomingEntry), dst(dst) {
      assert(srcs != nullptr);
      assert(incomingEntry != nullptr);
      assert(dst != nullptr);
    }
  };

  struct NodeControlBlock {
    friend LinkedGraph;

    ref_count external_count;
    ref_count live_parent_count;
    V payload;
    ControlBlock *controlBlock;
    EdgeList *outgoing;
    EdgeList *incoming; // storage.

    template <typename... Args>
    NodeControlBlock(ControlBlock *controlBlock, Args &&...args)
        : external_count(1), live_parent_count(0),
          payload(std::forward<Args>(args)...), controlBlock(controlBlock),
          outgoing(nullptr), incoming(nullptr) {}

    void decExternalCount() {
      assert(external_count > 0);
      external_count -= 1;
      cascadeLifeness();
    }

    void decLiveParentCount() {
      assert(live_parent_count > 0);
      live_parent_count -= 1;
      cascadeLifeness();
    }

    void cascadeLifeness() {
      if (external_count > 0 || live_parent_count > 0) {
        return;
      }
      OutgoingNodeList queue{nullptr, nullptr, nullptr};
      OutgoingNodeList *tail = &queue;
      ControlBlock *cb = controlBlock;

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

    static void enqueueDeadChildren(NodeControlBlock *node,
                                    OutgoingNodeList *&queue) {
      assert(queue != nullptr);
      // NOTE: Dead nodes cannot have incoming edges!!
      assert(node->incoming == nullptr);

      // 1. Remove all outgoing edges, while accumulating a list of children.
      EdgeList *curr = node->outgoing;
      if (curr != nullptr) {
        const void *outgoingHead = static_cast<const void *>(node->outgoing);
        do {
          EdgeInfo *edgeInfo = curr->info;
          // Remove all outgoing edges from srcs nodes, while skipping self.
          OutgoingNodeList *srcs = edgeInfo->srcs;
          while (srcs != nullptr) {
            if (srcs->node != node) {
              EdgeList *outgoingEntry = srcs->outgoingEntry;
              assert(outgoingEntry != nullptr);
              if (outgoingEntry->prev == outgoingEntry) {
                assert(outgoingEntry->next == outgoingEntry);
                // node.outgoing list has size 1! (cyclic list)
                // => collapse to nullptr.
                srcs->node->outgoing = nullptr;
              } else {
                outgoingEntry->prev->next = outgoingEntry->next;
                outgoingEntry->next->prev = outgoingEntry->prev;
                if (outgoingEntry == srcs->node->outgoing) {
                  // remount head.
                  srcs->node->outgoing = outgoingEntry->next;
                }
              }
              node->controlBlock->destroyEdgeList(outgoingEntry);
            }
            OutgoingNodeList *srcNext = srcs->next;
            node->controlBlock->destroyOutgoingNodeList(srcs);
            srcs = srcNext;
          }
          // Remove all incoming edge from dst node
          NodeControlBlock *dst = edgeInfo->dst;
          EdgeList *dstIncoming = edgeInfo->incomingEntry;
          assert(dstIncoming != nullptr);
          if (dstIncoming->prev == dstIncoming) {
            assert(dstIncoming->next == dstIncoming);
            // node.incoming has list size 1! (cyclic list)
            // => collapse to nullptr
            dst->incoming = nullptr;
          } else {
            dstIncoming->prev->next = dstIncoming->next;
            dstIncoming->next->prev = dstIncoming->prev;
            if (dst->incoming == dstIncoming) {
              // remount head.
              dst->incoming = dstIncoming->next;
            }
          }
          node->controlBlock->destroyEdgeInfo(dstIncoming->info);
          node->controlBlock->destroyEdgeList(dstIncoming);

          EdgeList *next = curr->next;
          assert(dst->live_parent_count > 0);
          dst->live_parent_count -= 1;
          if (dst->external_count == 0 && dst->live_parent_count == 0) {
            curr->~EdgeList();
            // NOTE: Super evil memory reuse trick
            OutgoingNodeList *temp = reinterpret_cast<OutgoingNodeList *>(curr);
            new (temp) OutgoingNodeList(nullptr, nullptr, dst);
            // push to linked list!
            queue->next = temp;
            queue = temp;
          } else {
            node->controlBlock->destroyEdgeList(curr);
          }
          curr = next;
        } while (static_cast<const void *>(curr) != outgoingHead);
      }
      node->outgoing = nullptr;
    }
  };

  struct OutgoingNodeList {
    OutgoingNodeList *next;
    EdgeList *outgoingEntry;
    NodeControlBlock *node;

    OutgoingNodeList() = default;
    OutgoingNodeList(OutgoingNodeList *next, EdgeList *outgoingEntry,
                     NodeControlBlock *node)
        : next(next), outgoingEntry(outgoingEntry), node(node) {}
  };

  struct EdgeList {
    EdgeList *next;
    EdgeList *prev;
    EdgeInfo *info;

    EdgeList() = default;
    EdgeList(EdgeList *next, EdgeList *prev, EdgeInfo *info)
        : next(next), prev(prev), info(info) {}
  };
  static_assert(sizeof(OutgoingNodeList) == sizeof(EdgeList));

  struct ControlBlock {
    monotonic_pool_allocator<sizeof(EdgeList), alignof(EdgeList)> linkedPool;

    monotonic_pool_allocator<sizeof(NodeControlBlock),
                             alignof(NodeControlBlock)>
        nodePool;

    monotonic_pool_allocator<sizeof(EdgeInfo), alignof(EdgeInfo)> edgePool;

    NodeControlBlock *allocNode() {
      return static_cast<NodeControlBlock *>(nodePool.allocate(
          sizeof(NodeControlBlock), alignof(NodeControlBlock)));
    }
    void destroyNode(NodeControlBlock *nodeBlock) {
      assert(nodeBlock != nullptr);
      assert(nodeBlock->external_count == 0);
      assert(nodeBlock->live_parent_count == 0);
      assert(nodeBlock->incoming == nullptr);
      if constexpr (!std::is_trivially_destructible_v<NodeControlBlock>) {
        nodeBlock->~NodeControlBlock();
      }
      nodePool.deallocate(nodeBlock);
    }

    EdgeInfo *allocEdgeInfo() {
      return static_cast<EdgeInfo *>(
          edgePool.allocate(sizeof(EdgeInfo), alignof(EdgeInfo)));
    }

    void destroyEdgeInfo(EdgeInfo *ptr) {
      assert(ptr != nullptr);
      if constexpr (!std::is_trivially_destructible_v<EdgeInfo>) {
        ptr->~EdgeInfo();
      }
      edgePool.deallocate(ptr);
    }

    EdgeList *allocEdgeList() {
      return static_cast<EdgeList *>(
          linkedPool.allocate(sizeof(EdgeList), alignof(EdgeList)));
    }

    void destroyEdgeList(EdgeList *ptr) {
      assert(ptr != nullptr);
      static_assert(std::is_trivially_destructible_v<EdgeList>);
      static_assert(sizeof(OutgoingNodeList) == sizeof(EdgeList));
      static_assert(alignof(OutgoingNodeList) == alignof(EdgeList));
      linkedPool.deallocate(ptr);
    }

    OutgoingNodeList *allocOutgoingNodeList() {
      return static_cast<OutgoingNodeList *>(linkedPool.allocate(
          sizeof(OutgoingNodeList), alignof(OutgoingNodeList)));
    }

    void destroyOutgoingNodeList(OutgoingNodeList *ptr) {
      assert(ptr != nullptr);
      static_assert(std::is_trivially_destructible_v<OutgoingNodeList>);
      static_assert(sizeof(OutgoingNodeList) == sizeof(EdgeList));
      static_assert(alignof(OutgoingNodeList) == alignof(EdgeList));
      linkedPool.deallocate(ptr);
    }
  };

public:
  using Edge = const void *;

  void removeEdge(Edge edge) {
    EdgeInfo *edgeInfo = static_cast<EdgeInfo *>(edge);
    // 1. Remove outgoing from srcs, while deallocating.
    OutgoingNodeList *srcs = edgeInfo->srcs;
    while (srcs != nullptr) {
      EdgeList *outgoingEntry = srcs->outgoingEntry;
      assert(outgoingEntry != nullptr);
      if (outgoingEntry->prev == outgoingEntry) {
        assert(outgoingEntry->next == outgoingEntry);
        srcs->node->outgoing = nullptr;
      } else {
        outgoingEntry->prev->next = outgoingEntry->next;
        outgoingEntry->next->prev = outgoingEntry->prev;
        if (outgoingEntry == srcs->node->outgoing) {
          srcs->node->outgoing = outgoingEntry->next;
        }
      }
      m_controlBlock->destroyEdgeList(outgoingEntry);
      OutgoingNodeList *next = srcs->next;
      m_controlBlock->destroyOutgoingNodeList(srcs);
      srcs = next;
    }
    // 2. Remove incoming entry and deallocate.
    EdgeList *incomingEntry = edgeInfo->incomingEntry;
    if (incomingEntry->prev == incomingEntry) {
      assert(incomingEntry->next == incomingEntry);
      edgeInfo->dst->incoming = nullptr;
    } else {
      incomingEntry->prev->next = incomingEntry->next;
      incomingEntry->next->prev = incomingEntry->prev;
      if (incomingEntry == edgeInfo->dst->incoming) {
        edgeInfo->dst->incoming = incomingEntry->next;
      }
    }
    m_controlBlock->destroyEdgeList(incomingEntry);
    // 3. Decrement live_parent_count of dst.
    edgeInfo->dst->decLiveParentCount();
    m_controlBlock->destroyEdgeInfo(edgeInfo);
  }

  class Node {
  public:
    friend LinkedGraph;

    Node(const Node &o) : m_controlBlock(o.m_controlBlock) {
      if (m_controlBlock != nullptr) {
        m_controlBlock->external_count += 1;
      }
    }
    Node &operator=(const Node &o) {
      if (this == &o) {
        return *this;
      }
      release();
      m_controlBlock = o.m_controlBlock;
      if (m_controlBlock != nullptr) {
        m_controlBlock->external_count += 1;
      }
      return *this;
    }

    Node(Node &&o) : m_controlBlock(std::exchange(o.m_controlBlock, nullptr)) {}
    Node &operator=(Node &&o) {
      if (this == &o) {
        return *this;
      }
      release();
      m_controlBlock = std::exchange(o.m_controlBlock, nullptr);
      return *this;
    }

    ~Node() { release(); }

    template <typename... Args>
      requires std::constructible_from<E, Args...>
    Edge addEdgeTo(const Node &dst, Args &&...args) {
      assert(m_controlBlock != nullptr);
      return LinkedGraph::addEdge_Internal(*this, dst,
                                           std::forward<Args>(args)...);
    }

    template <typename... Args>
      requires std::constructible_from<E, Args...>
    Edge addBinaryEdgeTo(const Node &src1, const Node &dst, Args &&...args) {
      assert(m_controlBlock != nullptr);
      return LinkedGraph::addBinaryEdge_Internal(*this, src1, dst,
                                                 std::forward<Args>(args)...);
    }

    auto incoming() {
      // TODO returns range of incoming edges.
    }

    auto outgoing() {
      // TODO returns range of outgoing edges.
    }

    auto graph() {
      // TODO returns reference to LinkedGraph.
    }

    void release() {
      if (m_controlBlock != nullptr) {
        m_controlBlock->decExternalCount();
        m_controlBlock = nullptr;
      }
    }

  private:
    explicit Node(NodeControlBlock *cb) : m_controlBlock(cb) {}
    NodeControlBlock *m_controlBlock;
  };
  friend Node;

private:
  template <typename... Args>
    requires std::constructible_from<E, Args...>
  static Edge addEdge_Internal(const Node &src, const Node &dst,
                               Args &&...args) {
    assert(dst.m_controlBlock != nullptr);
    assert(src.m_controlBlock != nullptr);
    assert(src.m_controlBlock->controlBlock ==
           dst.m_controlBlock->controlBlock);
    // NOTE: self edges are not allowed!
    assert(src.m_controlBlock != dst.m_controlBlock);
    ControlBlock *cb = src.m_controlBlock->controlBlock;
    assert(cb != nullptr);

    EdgeInfo *edgeInfo = cb->allocEdgeInfo();

    NodeControlBlock *dstNode = dst.m_controlBlock;

    EdgeList *outgoingEntry = cb->allocEdgeList();
    new (outgoingEntry) EdgeList(nullptr, nullptr, edgeInfo);
    if (src.m_controlBlock->outgoing == nullptr) {
      outgoingEntry->prev = outgoingEntry;
      outgoingEntry->next = outgoingEntry;
      src.m_controlBlock->outgoing = outgoingEntry;
    } else {
      outgoingEntry->next = src.m_controlBlock->outgoing->next;
      outgoingEntry->prev = src.m_controlBlock->outgoing;
      src.m_controlBlock->outgoing->next->prev = outgoingEntry;
      src.m_controlBlock->outgoing->next = outgoingEntry;
    }
    OutgoingNodeList *srcs = cb->allocOutgoingNodeList();
    new (srcs) OutgoingNodeList(nullptr, outgoingEntry, src.m_controlBlock);

    EdgeList *incomingEntry = cb->allocEdgeList();
    new (incomingEntry) EdgeList(nullptr, nullptr, edgeInfo);
    if (dstNode->incoming == nullptr) {
      incomingEntry->next = incomingEntry;
      incomingEntry->prev = incomingEntry;
      dstNode->incoming = incomingEntry;
    } else {
      incomingEntry->next = dstNode->incoming->next;
      incomingEntry->prev = dstNode->incoming;
      dstNode->incoming->next->prev = incomingEntry;
      dstNode->incoming->next = incomingEntry;
    }

    new (edgeInfo) EdgeInfo(srcs, incomingEntry, dst.m_controlBlock,
                            std::forward<Args>(args)...);
    dstNode->live_parent_count += 1;
    return static_cast<const void *>(edgeInfo);
  }

  template <typename... Args>
    requires std::constructible_from<E, Args...>
  static Edge addBinaryEdge_Internal(const Node &src0, const Node &src1,
                                     const Node &dst, Args &&...args) {
    assert(dst.m_controlBlock != nullptr);
    assert(src0.m_controlBlock != nullptr);
    assert(src1.m_controlBlock != nullptr);
    // NOTE: self edges are not allowed!
    assert(src0.m_controlBlock != dst.m_controlBlock);
    assert(src1.m_controlBlock != dst.m_controlBlock);
    assert(src0.m_controlBlock != src1.m_controlBlock);

    ControlBlock *cb = src0.m_controlBlock->controlBlock;
    assert(cb != nullptr);

    EdgeInfo *edgeInfo = cb->allocEdgeInfo();

    NodeControlBlock *dstNode = dst.m_controlBlock;

    EdgeList *outgoingEntrySrc0 = cb->allocEdgeList();
    new (outgoingEntrySrc0) EdgeList(nullptr, nullptr, edgeInfo);
    if (src0.m_controlBlock->outgoing == nullptr) {
      outgoingEntrySrc0->prev = outgoingEntrySrc0;
      outgoingEntrySrc0->next = outgoingEntrySrc0;
      src0.m_controlBlock->outgoing = outgoingEntrySrc0;
    } else {
      outgoingEntrySrc0->next = src0.m_controlBlock->outgoing->next;
      outgoingEntrySrc0->prev = src0.m_controlBlock->outgoing;
      src0.m_controlBlock->outgoing->next->prev = outgoingEntrySrc0;
      src0.m_controlBlock->outgoing->next = outgoingEntrySrc0;
    }

    EdgeList *outgoingEntrySrc1 = cb->allocEdgeList();
    new (outgoingEntrySrc1) EdgeList(nullptr, nullptr, edgeInfo);
    if (src1.m_controlBlock->outgoing == nullptr) {
      outgoingEntrySrc1->prev = outgoingEntrySrc1;
      outgoingEntrySrc1->next = outgoingEntrySrc1;
      src1.m_controlBlock->outgoing = outgoingEntrySrc1;
    } else {
      outgoingEntrySrc1->next = src1.m_controlBlock->outgoing->next;
      outgoingEntrySrc1->prev = src1.m_controlBlock->outgoing;
      src1.m_controlBlock->outgoing->next->prev = outgoingEntrySrc1;
      src1.m_controlBlock->outgoing->next = outgoingEntrySrc1;
    }

    OutgoingNodeList *srcs0 = cb->allocOutgoingNodeList();
    OutgoingNodeList *srcs1 = cb->allocOutgoingNodeList();

    new (srcs0) OutgoingNodeList(srcs1, outgoingEntrySrc0, src0.m_controlBlock);
    new (srcs1)
        OutgoingNodeList(nullptr, outgoingEntrySrc1, src1.m_controlBlock);

    EdgeList *incomingEntry = cb->allocEdgeList();
    new (incomingEntry) EdgeList(nullptr, nullptr, edgeInfo);
    if (dstNode->incoming == nullptr) {
      incomingEntry->next = incomingEntry;
      incomingEntry->prev = incomingEntry;
      dstNode->incoming = incomingEntry;
    } else {
      incomingEntry->next = dstNode->incoming->next;
      incomingEntry->prev = dstNode->incoming;
      dstNode->incoming->next->prev = incomingEntry;
      dstNode->incoming->next = incomingEntry;
    }

    new (edgeInfo) EdgeInfo(srcs0, incomingEntry, dst.m_controlBlock,
                            std::forward<Args>(args)...);
    dstNode->live_parent_count += 1;
    return static_cast<const void *>(edgeInfo);
  }

public:
  template <typename... Args>
    requires std::constructible_from<E, Args...>
  static Edge addEdge(const Node &src, const Node &dst, Args &&...args) {
    return addEdge_Internal(src, dst, std::forward<Args>(args)...);
  }

  template <typename... Args>
    requires std::constructible_from<E, Args...>
  static Edge addBinaryEdge(const Node &src0, const Node &src1, const Node &dst,
                            Args &&...args) {
    return addBinaryEdge_Internal(src0, src1, dst, std::forward<Args>(args)...);
  }

  template <typename... Args> Node createNode(Args &&...args) {
    NodeControlBlock *nodeBlock = m_controlBlock->allocNode();
    new (nodeBlock)
        NodeControlBlock(m_controlBlock.get(), std::forward<Args>(args)...);
    return Node{nodeBlock};
  }

private:
  std::unique_ptr<ControlBlock> m_controlBlock;
};

} // namespace denox::memory
