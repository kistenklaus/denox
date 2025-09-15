#pragma once

#include "memory/allocator/mallocator.hpp"
#include <cassert>
#include <cstddef>
#include <fmt/format.h>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace denox::testing {

struct Allocation {
  std::size_t size;
  std::size_t align;
};

class ProfileAllocator {
private:
  struct ControlBlock {
    std::unordered_map<void *, Allocation> m_allocations;
    std::size_t m_peakAllocationCount = 0;
    std::size_t m_allocatedBytes = 0;
    std::size_t m_peakAllocatedBytes = 0;
    [[no_unique_address]] memory::mallocator m_upstream;
  };

public:
  ProfileAllocator() : m_controlBlock(std::make_shared<ControlBlock>()) {}

  void *allocate(std::size_t size, std::size_t align) {
    if (size == 0) {
      throw std::runtime_error(
          "Invalid allocation: Allocating with size 0 is invalid!");
    }
    if (align == 0) {
      throw std::runtime_error(
          "Invalid allocation: Allocating with align 0 is invalid!");
    }
    void *ptr = m_controlBlock->m_upstream.allocate(size, align);
    if (m_controlBlock->m_allocations.contains(ptr)) {
      throw std::runtime_error(
          "Invalid allocation: Returns the same pointer twice.");
    }
    m_controlBlock->m_allocations.emplace(ptr, Allocation{size, align});
    m_controlBlock->m_allocatedBytes += size;
    m_controlBlock->m_peakAllocatedBytes = std::max(m_controlBlock->m_peakAllocatedBytes, m_controlBlock->m_allocatedBytes);
    m_controlBlock->m_peakAllocationCount =
        std::max(m_controlBlock->m_peakAllocationCount, m_controlBlock->m_allocations.size());
    return ptr;
  }

  void deallocate(void *ptr, std::size_t size, std::size_t align) {
    auto alloc = m_controlBlock->m_allocations.find(ptr);
    if (alloc == m_controlBlock->m_allocations.end()) {
      throw std::runtime_error("Invalid deallocation: Pointer does not exist.");
    }
    if (alloc->second.size != size) {
      throw std::runtime_error(fmt::format(
          "Invalid deallocation: size of allocation and "
          "deallocation must match! Allocated {} Bytes, Deallocated {} Bytes.",
          alloc->second.size, size));
    }
    if (alloc->second.align != align) {
      throw std::runtime_error("Invalid deallocation: align of allocation and "
                               "deallocation must match!");
    }
    m_controlBlock->m_allocatedBytes -= alloc->second.size;
    m_controlBlock->m_allocations.erase(alloc);
    m_controlBlock->m_upstream.deallocate(ptr);
  }

  void deallocate(void *ptr) {
    auto alloc = m_controlBlock->m_allocations.find(ptr);
    if (alloc == m_controlBlock->m_allocations.end()) {
      throw std::runtime_error("Invalid deallocation: Pointer does not exist.");
    }
    m_controlBlock->m_allocatedBytes -= alloc->second.size;
    m_controlBlock->m_allocations.erase(alloc);
    m_controlBlock->m_upstream.deallocate(ptr);
  }

  std::size_t allocationCount() { return m_controlBlock->m_allocations.size(); }

  std::size_t allocatedBytes() { return m_controlBlock->m_allocatedBytes; }

  std::size_t peakAllocatedBytes() { return m_controlBlock->m_peakAllocatedBytes; }

  std::size_t peakAllocationCount() { return m_controlBlock->m_peakAllocationCount; }

private:
  std::shared_ptr<ControlBlock> m_controlBlock;
};

} // namespace denox::testing
