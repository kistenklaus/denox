#pragma once

#include "memory/allocator/monotone_pool_allocator.hpp"
#include "memory/container/span.hpp"
#include "memory/hypergraph/NullWeight.hpp"
#include <cassert>
#include <iterator>
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
                      NodeControlBlock *dst, Args &&...args) noexcept
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
    NodeControlBlock(ControlBlock *controlBlock, Args &&...args) noexcept
        : external_count(1), live_parent_count(0),
          payload(std::forward<Args>(args)...), controlBlock(controlBlock),
          outgoing(nullptr), incoming(nullptr) {}

    void decExternalCount() noexcept {
      assert(external_count > 0);
      external_count -= 1;
      cascadeLifeness();
    }

    void decLiveParentCount() noexcept {
      assert(live_parent_count > 0);
      live_parent_count -= 1;
      cascadeLifeness();
    }

    void cascadeLifeness() noexcept {
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
                                    OutgoingNodeList *&queue) noexcept {
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

    OutgoingNodeList() noexcept = default;
    OutgoingNodeList(OutgoingNodeList *next, EdgeList *outgoingEntry,
                     NodeControlBlock *node) noexcept
        : next(next), outgoingEntry(outgoingEntry), node(node) {}
  };

  struct EdgeList {
    EdgeList *next;
    EdgeList *prev;
    EdgeInfo *info;

    EdgeList() noexcept = default;
    EdgeList(EdgeList *next, EdgeList *prev, EdgeInfo *info) noexcept
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
    void destroyNode(NodeControlBlock *nodeBlock) noexcept {
      assert(nodeBlock != nullptr);
      assert(nodeBlock->external_count == 0);
      assert(nodeBlock->live_parent_count == 0);
      assert(nodeBlock->incoming == nullptr);
      if constexpr (!std::is_trivially_destructible_v<NodeControlBlock>) {
        nodeBlock->~NodeControlBlock();
      }
      nodePool.deallocate(nodeBlock);
    }

    EdgeInfo *allocEdgeInfo() noexcept {
      return static_cast<EdgeInfo *>(
          edgePool.allocate(sizeof(EdgeInfo), alignof(EdgeInfo)));
    }

    void destroyEdgeInfo(EdgeInfo *ptr) noexcept {
      assert(ptr != nullptr);
      if constexpr (!std::is_trivially_destructible_v<EdgeInfo>) {
        ptr->~EdgeInfo();
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
  struct Edge {
  private:
    friend LinkedGraph;

    Edge(EdgeInfo *info) noexcept : m_info(info) {}
    EdgeInfo *m_info;
  };

  class Node {
  public:
    friend LinkedGraph;

    Node(const Node &o) noexcept : m_controlBlock(o.m_controlBlock) {
      if (m_controlBlock != nullptr) {
        m_controlBlock->external_count += 1;
      }
    }
    Node &operator=(const Node &o) noexcept {
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

    Node(Node &&o) noexcept
        : m_controlBlock(std::exchange(o.m_controlBlock, nullptr)) {}
    Node &operator=(Node &&o) noexcept {
      if (this == &o) {
        return *this;
      }
      release();
      m_controlBlock = std::exchange(o.m_controlBlock, nullptr);
      return *this;
    }

    ~Node() noexcept { release(); }

    class EdgeIt {
    public:
      friend Node;
      using iterator_category = std::input_iterator_tag;
      using iterator_concept = std::input_iterator_tag;
      using difference_type = std::ptrdiff_t;
      using value_type = Edge;
      using reference = Edge;

      EdgeIt() noexcept : m_head(nullptr), m_curr(nullptr) {}

      reference operator*() const noexcept {
        assert(m_curr != nullptr);
        return Edge{m_curr->info};
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

    struct IncomingList {
    public:
      friend Node;

      EdgeIt erase(EdgeIt pos) noexcept {
        assert(pos.m_head == &m_node.m_controlBlock->incoming);
        // NOTE: Because we hold a external_count of m_node, we are guaranteed
        // that this operation will never cascade!
        // Beccause of this just going ++next is actually valid, otherwise
        // next might point to a edge which was deleted in a cascade!
        EdgeIt next = pos;
        ++next;
        LinkedGraph::removeEdge_Internal(m_node.m_controlBlock->controlBlock,
                                         *pos);
        return next;
      }

      template <typename... Args>
      EdgeIt insert_after_with_dynamic_srcs(EdgeIt pos,
                                            memory::span<const Node *> srcs,
                                            Args &&...args) noexcept {
        const Node &dst = m_node;

        NodeControlBlock *dstNode = dst.m_controlBlock;
        ControlBlock *cb = dstNode->controlBlock;
        assert(cb != nullptr);
        assert(pos.m_head == &dstNode->incoming);
        assert(dstNode->controlBlock == cb);
        assert(!srcs.empty());

        EdgeInfo *edgeInfo = cb->allocEdgeInfo();

        OutgoingNodeList *srcsList = nullptr;
        for (int s = static_cast<int>(srcs.size()) - 1; s >= 0; --s) {
          const Node *ptr = srcs[s];
          assert(ptr && ptr->m_controlBlock &&
                 ptr->m_controlBlock->controlBlock == cb);
          assert(ptr->m_controlBlock != dstNode);
          const Node &src = *srcs[s];
          EdgeList *outgoingEntry = cb->allocEdgeList();
          new (outgoingEntry) EdgeList(nullptr, nullptr, edgeInfo);
          auto &srcNode = src.m_controlBlock;
          assert(srcNode->controlBlock == cb);
          if (srcNode->outgoing == nullptr) {
            outgoingEntry->prev = outgoingEntry;
            outgoingEntry->next = outgoingEntry;
            srcNode->outgoing = outgoingEntry;
          } else {
            outgoingEntry->next = srcNode->outgoing->next;
            outgoingEntry->prev = srcNode->outgoing;
            srcNode->outgoing->next->prev = outgoingEntry;
            srcNode->outgoing->next = outgoingEntry;
          }
          OutgoingNodeList *srcsEntry = cb->allocOutgoingNodeList();
          new (srcsEntry) OutgoingNodeList(srcsList, outgoingEntry, srcNode);
          srcsList = srcsEntry;
        }

        EdgeList *incomingEntry = cb->allocEdgeList();
        new (incomingEntry) EdgeList(nullptr, nullptr, edgeInfo);
        if (dstNode->incoming == nullptr) {
          assert(pos == end()); // must be empty!
          incomingEntry->next = incomingEntry;
          incomingEntry->prev = incomingEntry;
          dstNode->incoming = incomingEntry;
          // NOTE: ++pos, now points to incomingEntry!,
          // recovers m_curr == nullptr, by reloading the head!
        } else {
          assert(pos.m_curr != nullptr);
          incomingEntry->next = pos.m_curr->next;
          incomingEntry->prev = pos.m_curr;
          pos.m_curr->next->prev = incomingEntry;
          pos.m_curr->next = incomingEntry;
        }

        new (edgeInfo) EdgeInfo(srcsList, incomingEntry, dstNode,
                                std::forward<Args>(args)...);
        dstNode->live_parent_count += 1;
        return ++pos;
      }

      template <typename... Args>
      EdgeIt insert_after(EdgeIt pos, const Node &src, Args &&...args) noexcept {
        return insert_after_with_dynamic_srcs(pos, memory::span{&src, 1},
                                              std::forward<Args>(args)...);
      }

      template <typename... Args>
      EdgeIt insert_after(EdgeIt pos, const Node &src0, const Node &src1,
                          Args &&...args) noexcept {
        const Node *srcs[] = {&src0, &src1};
        return insert_after_with_dynamic_srcs(pos, srcs,
                                              std::forward<Args>(args)...);
      }

      template <typename... Args>
      EdgeIt insert(const Node &src0, const Node &src1, Args &&...args) noexcept {
        const Node *srcs[] = {&src0, &src1};
        return insert_after_with_dynamic_srcs(begin(), srcs,
                                              std::forward<Args>(args)...);
      }
      template <typename... Args>
      EdgeIt insert(const Node &src, Args &&...args) noexcept {
        return insert_after_with_dynamic_srcs(begin(), memory::span{&src, 1},
                                              std::forward<Args>(args)...);
      }

      [[nodiscard]] EdgeIt begin() noexcept {
        return EdgeIt(&m_node.m_controlBlock->incoming,
                      m_node.m_controlBlock->incoming);
      }
      [[nodiscard]] EdgeIt end() noexcept {
        return EdgeIt(&m_node.m_controlBlock->incoming, nullptr);
      }

    private:
      explicit IncomingList(Node node) noexcept : m_node(std::move(node)) {}
      Node m_node; // <- holds external_count!
    };
    friend IncomingList;

    [[nodiscard]] IncomingList incoming() const noexcept { return IncomingList(*this); }

    struct OutgoingList {
    public:
      friend Node;

      EdgeIt erase(EdgeIt pos) noexcept {
        assert(m_node.m_controlBlock != nullptr);
        assert(pos.m_head == &m_node.m_controlBlock->outgoing);
        assert(pos.m_curr != nullptr); // <- cannot erase from an empty list!
        // 1. Mark as a sentinel!
        // NOTE: outgoing does not have ownership,
        // incoming holds ownership of EdgeInfo.
        auto cb = m_node.m_controlBlock->controlBlock;
        assert(cb != nullptr);
        EdgeList *entry = pos.m_curr;
        {
          EdgeInfo *info = entry->info;
          entry->info = nullptr;
          LinkedGraph::removeEdge_Internal(cb, Edge(info));
        }
        // Still a sentinal, nobody touched it.
        assert(entry->info == nullptr);
        if (entry->prev == entry) {
          assert(entry->next == entry);
          m_node.m_controlBlock->outgoing = nullptr;
          cb->destroyEdgeList(entry);
          return end();
        } else {
          ++pos;
          EdgeList *next = entry->next;
          EdgeList *prev = entry->prev;
          prev->next = next;
          next->prev = prev;
          if (entry == m_node.m_controlBlock->outgoing) {
            m_node.m_controlBlock->outgoing = next;
          }
          cb->destroyEdgeList(entry);
          return pos;
        }
      }

      template <typename... Args>
      EdgeIt insert_after_with_dynamic_srcs(
          EdgeIt pos,
          std::span<const Node *> additionalSources, // <- can be empty.
          const Node &dst, Args &&...args) noexcept {
        assert(m_node.m_controlBlock != nullptr);
        assert(pos.m_head == &m_node.m_controlBlock->outgoing);
        auto cb = m_node.m_controlBlock->controlBlock;
        const Node &src0 = m_node;
        const auto src0Node = src0.m_controlBlock;

        assert(dst.m_controlBlock != nullptr);
        NodeControlBlock *dstNode = dst.m_controlBlock;
        assert(dstNode->controlBlock == cb);
        assert(dstNode != src0Node);

        EdgeInfo *edgeInfo = cb->allocEdgeInfo();

        OutgoingNodeList *srcsList = nullptr;
        for (int s = static_cast<int>(additionalSources.size()) - 1; s >= 0;
             --s) {
          const Node *ptr = additionalSources[static_cast<std::size_t>(s)];
          assert(ptr && ptr->m_controlBlock &&
                 ptr->m_controlBlock->controlBlock == cb);
          assert(ptr->m_controlBlock != dstNode);
          const Node &src = *ptr;
          EdgeList *outgoingEntry = cb->allocEdgeList();
          new (outgoingEntry) EdgeList(nullptr, nullptr, edgeInfo);
          auto &srcNode = src.m_controlBlock;
          assert(srcNode->controlBlock == cb);
          if (srcNode->outgoing == nullptr) {
            outgoingEntry->prev = outgoingEntry;
            outgoingEntry->next = outgoingEntry;
            srcNode->outgoing = outgoingEntry;
          } else {
            outgoingEntry->next = srcNode->outgoing->next;
            outgoingEntry->prev = srcNode->outgoing;
            srcNode->outgoing->next->prev = outgoingEntry;
            srcNode->outgoing->next = outgoingEntry;
          }
          OutgoingNodeList *srcsEntry = cb->allocOutgoingNodeList();
          new (srcsEntry) OutgoingNodeList(srcsList, outgoingEntry, srcNode);
          srcsList = srcsEntry;
        }
        EdgeList *outgoingEntry = cb->allocEdgeList();
        new (outgoingEntry) EdgeList(nullptr, nullptr, edgeInfo);
        if (src0Node->outgoing == nullptr) {
          assert(pos == end()); // <- empty list.
          outgoingEntry->prev = outgoingEntry;
          outgoingEntry->next = outgoingEntry;
          src0Node->outgoing = outgoingEntry;
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
        return ++pos;
      }

      template <typename... Args>
      EdgeIt insert_after(EdgeIt pos, const Node &dst, Args &&...args) noexcept {
        return insert_after_with_dynamic_srcs(
            pos, memory::span<const Node *>(), dst,
            std::forward<Args>(args)...);
      }

      template <typename... Args>
      EdgeIt insert_after(EdgeIt pos, const Node &src1, const Node &dst,
                          Args &&...args) noexcept {
        return insert_after_with_dynamic_srcs(
            pos, memory::span<const Node *>{&src1, 1}, dst,
            std::forward<Args>(args)...);
      }

      /// Inserts anywhere
      template <typename... Args> EdgeIt insert(const Node &dst, Args &&...args) noexcept {
        return insert_after_with_dynamic_srcs(
            begin(), memory::span<const Node *>(), dst,
            std::forward<Args>(args)...);
      }

      /// Inserts anywhere
      template <typename... Args>
      EdgeIt insert(const Node &src1, const Node &dst, Args &&...args) noexcept {
        return insert_after_with_dynamic_srcs(
            begin(), memory::span<const Node *>(&src1, 1), dst,
            std::forward<Args>(args)...);
      }

      [[nodiscard]] EdgeIt begin() noexcept {
        return EdgeIt(&m_node.m_controlBlock->outgoing,
                      m_node.m_controlBlock->outgoing);
      }

      [[nodiscard]] EdgeIt end() noexcept {
        return EdgeIt(&m_node.m_controlBlock->outgoing, nullptr);
      }

    private:
      explicit OutgoingList(Node node) : m_node(std::move(node)) {}
      Node m_node;
    };

    [[nodiscard]] OutgoingList outgoing() const noexcept { return OutgoingList(*this); }

    void release() noexcept {
      if (m_controlBlock != nullptr) {
        m_controlBlock->decExternalCount();
        m_controlBlock = nullptr;
      }
    }

  private:
    explicit Node(NodeControlBlock *cb) noexcept : m_controlBlock(cb) {}
    NodeControlBlock *m_controlBlock;
  };
  friend Node;

private:
  static void removeEdge_Internal(ControlBlock *controlBlock, Edge edge) noexcept {
    EdgeInfo *edgeInfo = edge.m_info;
    // 1. Remove outgoing from srcs, while deallocating.
    OutgoingNodeList *srcs = edgeInfo->srcs;
    while (srcs != nullptr) {
      EdgeList *outgoingEntry = srcs->outgoingEntry;
      assert(outgoingEntry != nullptr);
      // NOTE: info == nullptr, means this is a node sentinal
      // ignore it removal is handled somewhere else!
      if (outgoingEntry->info != nullptr) {
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
        controlBlock->destroyEdgeList(outgoingEntry);
      }
      OutgoingNodeList *next = srcs->next;
      controlBlock->destroyOutgoingNodeList(srcs) ;
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
    controlBlock->destroyEdgeList(incomingEntry);
    // 3. Decrement live_parent_count of dst.
    edgeInfo->dst->decLiveParentCount();
    controlBlock->destroyEdgeInfo(edgeInfo);
  }

public:
  template <typename... Args> Node createNode(Args &&...args) noexcept {
    NodeControlBlock *nodeBlock = m_controlBlock->allocNode();
    new (nodeBlock)
        NodeControlBlock(m_controlBlock.get(), std::forward<Args>(args)...);
    return Node{nodeBlock};
  }

  LinkedGraph() : m_controlBlock(std::make_unique<ControlBlock>()) {}

private:
  std::unique_ptr<ControlBlock> m_controlBlock;
};

} // namespace denox::memory
