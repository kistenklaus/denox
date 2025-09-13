#pragma once

#include "memory/allocator/mallocator.hpp"
#include <cassert>
#include <cstddef>
#include <ratio>
#include <utility>
namespace denox::memory {

template <std::size_t BlockSize, std::size_t BlockAlign,
          typename Allocator = Mallocator,
          typename GrowthFactor = std::ratio<2, 1>>
class monotonic_pool_allocator {
public:
  static_assert(GrowthFactor::num >= GrowthFactor::den);
  using Self =
      monotonic_pool_allocator<BlockSize, BlockAlign, Allocator, GrowthFactor>;
  using upstream = Allocator;
  static constexpr std::size_t block_size = std::max(BlockSize, BlockAlign);
  static constexpr std::size_t block_align = BlockAlign;

  explicit monotonic_pool_allocator(std::size_t capacity = 0,
                                    const upstream &upstream = {})
      : m_upstream(upstream), m_buffer(nullptr), m_freelist(nullptr) {
    if (capacity != 0) {
      allocBlock(capacity + 1);
    }
  }

  monotonic_pool_allocator(const monotonic_pool_allocator &) = delete;
  monotonic_pool_allocator &
  operator=(const monotonic_pool_allocator &) = delete;

  monotonic_pool_allocator(monotonic_pool_allocator &&o)
      : m_upstream(std::move(o.m_upstream)),
        m_buffer(std::exchange(o.m_buffer, nullptr)),
        m_freelist(std::exchange(o.m_freelist, nullptr)) {}

  monotonic_pool_allocator &operator=(monotonic_pool_allocator &&o) {
    if (this == &o) {
      return *this;
    }
    release();
    m_upstream = std::move(o.m_upstream);
    m_buffer = std::exchange(o.m_buffer, nullptr);
    m_freelist = std::exchange(o.m_freelist, nullptr);
    return *this;
  }

  ~monotonic_pool_allocator() { release(); }

  void *allocate(std::size_t size, std::size_t align) {
    assert(size != 0);
    assert(size <= block_size);
    assert((BlockAlign % align) == 0);
    if (m_freelist == nullptr) {
      grow();
    }
    return popFreelist();
  }

  void deallocate(void *ptr) { pushFreelist(ptr); }

private:
  union Node {
    struct {
      Node *next;
    } free;
    struct {
      Node *next;
      std::size_t blockSize;
    } block;
    alignas(BlockAlign) std::byte value[block_size];
  };

  void release() {
    Node *block = m_buffer;
    while (block != nullptr) {
      Node *nextBlock = block->block.next;
      m_upstream.deallocate(block, block->block.blockSize * sizeof(Node), alignof(Node));
      block = nextBlock;
    }
    m_buffer = nullptr;
    m_freelist = nullptr;
  }

  void *popFreelist() {
    void *ptr = static_cast<void *>(m_freelist);
    assert(m_freelist != nullptr);
    m_freelist = m_freelist->free.next;
    return ptr;
  }

  void pushFreelist(void *ptr) {
    Node *node = static_cast<Node *>(ptr);
    node->free.next = m_freelist;
    m_freelist = node;
  }

  void grow() {
    std::size_t nextBlockSize;
    if (m_buffer == nullptr) {
      nextBlockSize = 2;
    } else {
      nextBlockSize =
          (m_buffer->block.blockSize * GrowthFactor::num) / GrowthFactor::den;
    }
    allocBlock(nextBlockSize);
  }

  void allocBlock(std::size_t blockSize) {
    Node *block = static_cast<Node*>(m_upstream.allocate(blockSize * sizeof(Node), alignof(Node)));
    Node *header = block;
    Node *const begin = header + 1;
    Node *const end = header + blockSize;
    for (Node *current = begin; current < end - 1; ++current) {
      current->free.next = current + 1;
    }
    (end - 1)->free.next = m_freelist;
    m_freelist = begin;
    block->block.blockSize = blockSize;
    block->block.next = m_buffer;
    m_buffer = block;
  }

  [[no_unique_address]] upstream m_upstream;
  Node *m_buffer;   // fwd. linked list of blocks
  Node *m_freelist; // fwd. linked freelist
};

} // namespace denox::memory
