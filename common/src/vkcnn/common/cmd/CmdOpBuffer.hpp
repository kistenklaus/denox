#pragma once

#include "vkcnn/common/cmd/CmdOp.hpp"
#include "vkcnn/common/cmd/CmdOpActivation.hpp"
#include "vkcnn/common/cmd/CmdOpBarrier.hpp"
#include "vkcnn/common/cmd/CmdOpConv.hpp"
#include "vkcnn/common/cmd/CmdOpCopy.hpp"
#include "vkcnn/common/cmd/CmdOpPool.hpp"
#include "vkcnn/common/cmd/CmdOpUpsample.hpp"
#include "vkcnn/common/memory/align.hpp"
#include <cassert>
#include <cstdlib>
#include <memory>
#include <spdlog/common.h>
#include <utility>

namespace vkcnn {

// Monotone builder pattern for a CmdOpSchedule
class CmdOpBuffer {
public:
  CmdOpBuffer() : m_buffer(nullptr), m_capacity(0), m_size(0) {}

  CmdOpBuffer(const CmdOpBuffer &o) noexcept
      : m_buffer(malloc(o.m_size)), m_capacity(o.m_size), m_size(o.m_size) {
    copy_construct(m_buffer, o.m_buffer, o.m_size);
  }

  CmdOpBuffer &operator=(const CmdOpBuffer &o) noexcept {
    if (this == &o) {
      return *this;
    }
    destruct(m_buffer, m_size);
    m_size = 0;
    if (o.m_size > m_capacity) {
      grow(o.m_size);
    }
    copy_construct(m_buffer, o.m_buffer, o.m_size);
    m_size = o.m_size;
    return *this;
  }
  CmdOpBuffer(CmdOpBuffer &&o) noexcept
      : m_buffer(std::exchange(o.m_buffer, nullptr)),
        m_capacity(std::exchange(o.m_capacity, 0)),
        m_size(std::exchange(o.m_size, 0)) {}

  CmdOpBuffer &operator=(CmdOpBuffer &&o) noexcept {
    if (this == &o) {
      return *this;
    }
    destruct(m_buffer, m_size);
    m_size = 0;
    if (o.m_size > m_capacity) {
      grow(o.m_size);
    }
    move_construct(m_buffer, o.m_buffer, o.m_size);
    m_size = o.m_size;
    return *this;
  }

  void record(CmdOpConv cmd) {
    auto ptr = static_cast<CmdOpConv *>(allocate(CmdOp::Conv));
    new (ptr) CmdOpConv(std::move(cmd));
  }

  void record(CmdOpActivation cmd) {
    auto ptr = allocate(CmdOp::Activation);
    new (ptr) CmdOpActivation(std::move(cmd));
  }

  void record(CmdOpCopy cmd) {
    auto ptr = allocate(CmdOp::Copy);
    new (ptr) CmdOpCopy(std::move(cmd));
  }

  void record(CmdOpPool cmd) {
    auto ptr = allocate(CmdOp::Pool);
    new (ptr) CmdOpPool(std::move(cmd));
  }

  void record(CmdOpUpsample cmd) {
    auto ptr = allocate(CmdOp::Upsample);
    new (ptr) CmdOpUpsample(std::move(cmd));
  }

  void record(CmdOpBarrier cmd) {
    auto ptr = allocate(CmdOp::Barrier);
    new (ptr) CmdOpBarrier(std::move(cmd));
  }

  void release() {
    destruct(m_buffer, m_size);
    free(m_buffer);
    m_capacity = 0;
    m_size = 0;
  }

  void reset() { release(); }

  void shrink_to_fit() {
    if (m_capacity == m_size) {
      return;
    }
    void *newBuffer = malloc(m_size);
    move_construct(newBuffer, m_buffer, m_size);
    void *oldBuffer = std::exchange(m_buffer, newBuffer);
    destruct(oldBuffer, m_size);
    free(oldBuffer);
    m_capacity = m_size;
  }

  struct cmd_iterator;

  struct cmd {
    friend cmd_iterator;

    CmdOp op() const { return *m_op; }

    const CmdOpConv &Conv() const {
      assert(*m_op == CmdOp::Conv);
      return *reinterpret_cast<const CmdOpConv *>(m_cmd);
    }

    const CmdOpActivation &Activation() const {
      assert(*m_op == CmdOp::Activation);
      return *reinterpret_cast<const CmdOpActivation *>(m_cmd);
    }

    const CmdOpCopy &Copy() const {
      assert(*m_op == CmdOp::Copy);
      return *reinterpret_cast<const CmdOpCopy *>(m_cmd);
    }

    const CmdOpPool &Pool() const {
      assert(*m_op == CmdOp::Pool);
      return *reinterpret_cast<const CmdOpPool *>(m_cmd);
    }

    const CmdOpUpsample &Upsample() const {
      assert(*m_op == CmdOp::Upsample);
      return *reinterpret_cast<const CmdOpUpsample *>(m_cmd);
    }

    const CmdOpBarrier &Barrier() const {
      assert(*m_op == CmdOp::Barrier);
      return *reinterpret_cast<const CmdOpBarrier *>(m_cmd);
    }

  private:
    explicit cmd(const CmdOp *op, const std::byte *cmd)
        : m_op(op), m_cmd(cmd) {}
    const CmdOp *m_op;
    const std::byte *m_cmd;
  };

  struct cmd_ptr {
    friend cmd_iterator;
    const cmd &operator*() const { return m_cmd; }
    const cmd *operator->() const { return &m_cmd; }

  private:
    explicit cmd_ptr(cmd cmd) : m_cmd(cmd) {}
    cmd m_cmd;
  };

  struct cmd_iterator {
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = cmd;
    using pointer = cmd_ptr;
    using reference = cmd;

    cmd_iterator(const void *ptr)
        : m_ptr(static_cast<const std::byte *>(ptr)) {}

    reference operator*() const {
      const RecordHeader *header =
          reinterpret_cast<const RecordHeader *>(m_ptr);
      const std::byte *cmdPtr = align_up(m_ptr, header->op.alignment());
      return cmd{&header->op, cmdPtr};
    }
    pointer operator->() const {
      const RecordHeader *header =
          reinterpret_cast<const RecordHeader *>(m_ptr);
      const std::byte *cmdPtr = align_up(m_ptr, header->op.alignment());
      return cmd_ptr{cmd{&header->op, cmdPtr}};
    }

    cmd_iterator &operator++() {
      const RecordHeader *header =
          reinterpret_cast<const RecordHeader *>(m_ptr);
      const std::byte *cmdPtr =
          align_up(m_ptr + sizeof(RecordHeader), header->op.alignment());
      const std::byte *next = cmdPtr + header->op.size();
      m_ptr = align_up(next, alignof(RecordHeader));
      return *this;
    }

    cmd_iterator operator++(int) {
      cmd_iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    friend bool operator==(const cmd_iterator &a, const cmd_iterator &b) {
      return a.m_ptr == b.m_ptr;
    };
    friend bool operator!=(const cmd_iterator &a, const cmd_iterator &b) {
      return a.m_ptr != b.m_ptr;
    };

  private:
    const std::byte *m_ptr;
  };

  cmd_iterator begin() const { return cmd_iterator(m_buffer); }
  cmd_iterator end() const {
    return cmd_iterator(
        static_cast<const void *>(static_cast<const std::byte *>(m_buffer)));
  }

  std::size_t byteSize() const { return m_size; }

private:
  struct RecordHeader {
    CmdOp op;
  };

  void *allocate(CmdOp op) {
    std::size_t headerOffset = align_up(m_size, alignof(RecordHeader));
    std::size_t cmdOffset =
        align_up(headerOffset + sizeof(RecordHeader), op.alignment());
    std::size_t newSize = cmdOffset + op.size();

    if (newSize > m_capacity) {
      grow(newSize * 2);
    }
    assert(m_buffer != nullptr);
    std::byte *raw = static_cast<std::byte *>(m_buffer);
    RecordHeader *header = reinterpret_cast<RecordHeader *>(raw + headerOffset);
    std::construct_at(header, op);
    return static_cast<void *>(raw + cmdOffset);
  }

  static void copy_assign(void *dst, const void *src, std::size_t size) {
    std::size_t offset = 0;
    while (offset < size) {
      offset = align_up(offset, alignof(RecordHeader));
      auto srcPtr = static_cast<const std::byte *>(src) + offset;
      auto dstPtr = static_cast<std::byte *>(dst) + offset;

      auto srcHeader = reinterpret_cast<const RecordHeader *>(srcPtr);
      auto dstHeader = reinterpret_cast<RecordHeader *>(dstPtr);
      *dstHeader = *srcHeader;

      offset =
          align_up(offset + sizeof(RecordHeader), srcHeader->op.alignment());
      srcPtr = static_cast<const std::byte *>(src) + offset;
      dstPtr = static_cast<std::byte *>(dst) + offset;
      switch (srcHeader->op.tag()) {
      case CmdOpTag::Conv:
        *reinterpret_cast<CmdOpConv *>(dstPtr) =
            *reinterpret_cast<const CmdOpConv *>(srcPtr);
        break;
      case CmdOpTag::Activation:
        *reinterpret_cast<CmdOpActivation *>(dstPtr) =
            *reinterpret_cast<const CmdOpActivation *>(srcPtr);
        break;
      case CmdOpTag::Copy:
        *reinterpret_cast<CmdOpCopy *>(dstPtr) =
            *reinterpret_cast<const CmdOpCopy *>(srcPtr);
        break;
      case CmdOpTag::Pool:
        *reinterpret_cast<CmdOpPool *>(dstPtr) =
            *reinterpret_cast<const CmdOpPool *>(srcPtr);
        break;
      case CmdOpTag::Upsample:
        *reinterpret_cast<CmdOpUpsample *>(dstPtr) =
            *reinterpret_cast<const CmdOpUpsample *>(srcPtr);
        break;
      case CmdOpTag::Barrier:
        *reinterpret_cast<CmdOpBarrier *>(dstPtr) =
            *reinterpret_cast<const CmdOpBarrier *>(srcPtr);
        break;
      }
      offset += srcHeader->op.size();
    }
  }

  static void copy_construct(void *dst, const void *src, std::size_t size) {
    std::size_t offset = 0;
    while (offset < size) {
      offset = align_up(offset, alignof(RecordHeader));
      auto srcPtr = static_cast<const std::byte *>(src) + offset;
      auto dstPtr = static_cast<std::byte *>(dst) + offset;

      auto srcHeader = reinterpret_cast<const RecordHeader *>(srcPtr);
      auto dstHeader = reinterpret_cast<RecordHeader *>(dstPtr);
      new (dstHeader) RecordHeader(*srcHeader);

      offset =
          align_up(offset + sizeof(RecordHeader), srcHeader->op.alignment());
      srcPtr = static_cast<const std::byte *>(src) + offset;
      dstPtr = static_cast<std::byte *>(dst) + offset;
      switch (srcHeader->op.tag()) {
      case CmdOpTag::Conv: {
        using Op = CmdOpConv;
        new (dstPtr) Op(*reinterpret_cast<const Op *>(srcPtr));
        break;
      }
      case CmdOpTag::Activation: {
        using Op = CmdOpActivation;
        new (dstPtr) Op(*reinterpret_cast<const Op *>(srcPtr));
        break;
      }
      case CmdOpTag::Copy: {
        using Op = CmdOpCopy;
        new (dstPtr) Op(*reinterpret_cast<const Op *>(srcPtr));
        break;
      }
      case CmdOpTag::Pool: {
        using Op = CmdOpPool;
        new (dstPtr) Op(*reinterpret_cast<const Op *>(srcPtr));
        break;
      }
      case CmdOpTag::Upsample: {
        using Op = CmdOpUpsample;
        new (dstPtr) Op(*reinterpret_cast<const Op *>(srcPtr));
        break;
      }
      case CmdOpTag::Barrier: {
        using Op = CmdOpBarrier;
        new (dstPtr) Op(*reinterpret_cast<const Op *>(srcPtr));
        break;
      }
      }
      offset += srcHeader->op.size();
    }
  }

  static void move_assign(void *dst, const void *src, std::size_t size) {
    std::size_t offset = 0;
    while (offset < size) {
      offset = align_up(offset, alignof(RecordHeader));
      auto srcPtr = static_cast<const std::byte *>(src) + offset;
      auto dstPtr = static_cast<std::byte *>(dst) + offset;

      auto srcHeader = reinterpret_cast<const RecordHeader *>(srcPtr);
      auto dstHeader = reinterpret_cast<RecordHeader *>(dstPtr);
      *dstHeader = std::move(*srcHeader);

      offset =
          align_up(offset + sizeof(RecordHeader), srcHeader->op.alignment());
      srcPtr = static_cast<const std::byte *>(src) + offset;
      dstPtr = static_cast<std::byte *>(dst) + offset;
      switch (srcHeader->op.tag()) {
      case CmdOpTag::Conv:
        *reinterpret_cast<CmdOpConv *>(dstPtr) =
            std::move(*reinterpret_cast<const CmdOpConv *>(srcPtr));
        break;
      case CmdOpTag::Activation:
        *reinterpret_cast<CmdOpActivation *>(dstPtr) =
            std::move(*reinterpret_cast<const CmdOpActivation *>(srcPtr));
        break;
      case CmdOpTag::Copy:
        *reinterpret_cast<CmdOpCopy *>(dstPtr) =
            std::move(*reinterpret_cast<const CmdOpCopy *>(srcPtr));
        break;
      case CmdOpTag::Pool:
        *reinterpret_cast<CmdOpPool *>(dstPtr) =
            std::move(*reinterpret_cast<const CmdOpPool *>(srcPtr));
        break;
      case CmdOpTag::Upsample:
        *reinterpret_cast<CmdOpUpsample *>(dstPtr) =
            std::move(*reinterpret_cast<const CmdOpUpsample *>(srcPtr));
        break;
      case CmdOpTag::Barrier:
        *reinterpret_cast<CmdOpBarrier *>(dstPtr) =
            std::move(*reinterpret_cast<const CmdOpBarrier *>(srcPtr));
        break;
      }
      offset += srcHeader->op.size();
    }
  }

  static void move_construct(void *dst, const void *src, std::size_t size) {
    std::size_t offset = 0;
    while (offset < size) {
      offset = align_up(offset, alignof(RecordHeader));
      auto srcPtr = static_cast<const std::byte *>(src) + offset;
      auto dstPtr = static_cast<std::byte *>(dst) + offset;

      auto srcHeader = reinterpret_cast<const RecordHeader *>(srcPtr);
      auto dstHeader = reinterpret_cast<RecordHeader *>(dstPtr);
      new (dstHeader) RecordHeader(std::move(*srcHeader));

      offset =
          align_up(offset + sizeof(RecordHeader), srcHeader->op.alignment());
      srcPtr = static_cast<const std::byte *>(src) + offset;
      dstPtr = static_cast<std::byte *>(dst) + offset;
      switch (srcHeader->op.tag()) {
      case CmdOpTag::Conv: {
        using Op = CmdOpConv;
        new (dstPtr) Op(std::move(*reinterpret_cast<const Op *>(srcPtr)));
        break;
      }
      case CmdOpTag::Activation: {
        using Op = CmdOpActivation;
        new (dstPtr) Op(std::move(*reinterpret_cast<const Op *>(srcPtr)));
        break;
      }
      case CmdOpTag::Copy: {
        using Op = CmdOpCopy;
        new (dstPtr) Op(std::move(*reinterpret_cast<const Op *>(srcPtr)));
        break;
      }
      case CmdOpTag::Pool: {
        using Op = CmdOpPool;
        new (dstPtr) Op(std::move(*reinterpret_cast<const Op *>(srcPtr)));
        break;
      }
      case CmdOpTag::Upsample: {
        using Op = CmdOpUpsample;
        new (dstPtr) Op(std::move(*reinterpret_cast<const Op *>(srcPtr)));
        break;
      }
      case CmdOpTag::Barrier: {
        using Op = CmdOpBarrier;
        new (dstPtr) Op(std::move(*reinterpret_cast<const Op *>(srcPtr)));
        break;
      }
      }
      offset += srcHeader->op.size();
    }
  }

  static void destruct(void *buf, std::size_t size) {
    std::size_t offset = 0;
    while (offset < size) {
      offset = align_up(offset, alignof(RecordHeader));
      auto ptr = static_cast<std::byte *>(buf) + offset;

      auto header = reinterpret_cast<RecordHeader *>(ptr);

      offset = align_up(offset + sizeof(RecordHeader), header->op.alignment());
      ptr = static_cast<std::byte *>(buf) + offset;

      switch (header->op.tag()) {
      case CmdOpTag::Conv: {
        std::destroy_at(reinterpret_cast<CmdOpConv *>(ptr));
        break;
      }
      case CmdOpTag::Activation: {
        std::destroy_at(reinterpret_cast<CmdOpActivation *>(ptr));
        break;
      }
      case CmdOpTag::Copy: {
        std::destroy_at(reinterpret_cast<CmdOpCopy *>(ptr));
        break;
      }
      case CmdOpTag::Pool: {
        std::destroy_at(reinterpret_cast<CmdOpPool *>(ptr));
        break;
      }
      case CmdOpTag::Upsample: {
        std::destroy_at(reinterpret_cast<CmdOpUpsample *>(ptr));
        break;
      }
      case CmdOpTag::Barrier: {
        std::destroy_at(reinterpret_cast<CmdOpBarrier *>(ptr));
        break;
      }
      }
      offset += header->op.size();
    }
  }

  void grow(std::size_t newCapacity) {
    if (m_capacity >= newCapacity) {
      return; // <- we are monotone we never shrink!
    }
    void *newBuffer = malloc(newCapacity);
    move_construct(newBuffer, m_buffer, m_size);
    void *oldBuffer = std::exchange(m_buffer, newBuffer);
    destruct(oldBuffer, m_size);
    free(oldBuffer);
    m_capacity = newCapacity;
  }

  void *m_buffer;
  std::size_t m_capacity;
  std::size_t m_size;
};

} // namespace vkcnn
