#pragma once

#include "vkcnn/common/cmd/CmdOpBarrier.hpp"
#include "vkcnn/common/cmd/CmdOpBuffer.hpp"
#include "vkcnn/common/cmd/CmdOpDispatch.hpp"
#include "vkcnn/common/memory/align.hpp"
#include <cstring>
#include <iterator>

namespace vkcnn {


class CmdOpSchedule {
public:
  explicit CmdOpSchedule(const CmdOpBuffer &buffer) {
    std::size_t header_offset = 0;
    std::size_t op_offset =
        align_up(sizeof(CmdOpScheduleHeader), alignof(CmdOp));

    std::size_t cmdOT_offset = op_offset + sizeof(CmdOp) * buffer.m_op.size();
    cmdOT_offset = align_up(cmdOT_offset, alignof(std::uint64_t));

    std::size_t cmdBuf_offset =
        cmdOT_offset + sizeof(std::uint64_t) * buffer.m_cmdOT.size();
    cmdBuf_offset = align_up(cmdBuf_offset, alignof(std::uint64_t));

    std::size_t spirv_offset = cmdBuf_offset + buffer.m_cmdBuf.size();
    spirv_offset = align_up(spirv_offset, alignof(std::uint32_t));

    std::size_t weight_offset = spirv_offset + buffer.m_spirvBuf.size();
    weight_offset = align_up(weight_offset, alignof(double));

    std::size_t byteSize = weight_offset + buffer.m_paramBuf.size();

    m_buffer = malloc(byteSize);
    auto raw = static_cast<std::byte *>(m_buffer);
    new (raw + header_offset) CmdOpScheduleHeader{
        op_offset,    buffer.m_op.size(), cmdOT_offset, cmdBuf_offset,
        spirv_offset, weight_offset,      byteSize};

    std::memcpy(raw + op_offset, buffer.m_op.data(),
                buffer.m_op.size() * sizeof(CmdOp));
    std::memcpy(raw + cmdOT_offset, buffer.m_cmdOT.data(),
                buffer.m_cmdOT.size() * sizeof(std::uint64_t));
    std::memcpy(raw + cmdBuf_offset, buffer.m_cmdBuf.get(),
                buffer.m_cmdBuf.size());
    std::memcpy(raw + spirv_offset, buffer.m_spirvBuf.get(),
                buffer.m_spirvBuf.size());
    std::memcpy(raw + weight_offset, buffer.m_paramBuf.get(),
                buffer.m_paramBuf.size());
  }

  ~CmdOpSchedule() { release(); }
  CmdOpSchedule(const CmdOpSchedule &o) noexcept {
    auto oraw = static_cast<const std::byte *>(o.m_buffer);
    auto oheader = reinterpret_cast<const CmdOpScheduleHeader *>(oraw);
    std::size_t byteSize = oheader->byteSize;
    m_buffer = malloc(byteSize);
    std::memcpy(m_buffer, o.m_buffer, byteSize);
  }

  CmdOpSchedule &operator=(const CmdOpSchedule &o) noexcept {
    if (this == &o) {
      return *this;
    }
    release();
    auto oraw = static_cast<const std::byte *>(o.m_buffer);
    auto oheader = reinterpret_cast<const CmdOpScheduleHeader *>(oraw);
    std::size_t byteSize = oheader->byteSize;
    m_buffer = malloc(byteSize);
    std::memcpy(m_buffer, o.m_buffer, byteSize);
    return *this;
  }
  CmdOpSchedule(CmdOpSchedule &&o) noexcept
      : m_buffer(std::exchange(o.m_buffer, nullptr)) {}

  CmdOpSchedule &operator=(CmdOpSchedule &&o) noexcept {
    if (this == &o) {
      return *this;
    }
    release();
    m_buffer = std::exchange(o.m_buffer, nullptr);
    return *this;
  }

  void release() {
    if (m_buffer != nullptr) {
      free(m_buffer);
      m_buffer = nullptr;
    }
  }

  struct const_iterator;

  struct cmd {
  public:
    friend const_iterator;

    CmdOp op() const { return m_op; }

    const CmdOpDispatch &getDispatch() {
      assert(m_op == CmdOp::Dispatch);
      return *static_cast<const CmdOpDispatch *>(m_cmd);
    }

    const CmdOpBarrier &getBarrier() {
      assert(m_op == CmdOp::Barrier);
      return *static_cast<const CmdOpBarrier *>(m_cmd);
    }

  private:
    cmd(CmdOp op, void *cmd) : m_op(op), m_cmd(cmd) {}
    CmdOp m_op;
    const void *m_cmd;
  };

  struct cmd_ptr {
  public:
    friend const_iterator;
    const cmd *operator->() const { return &m_cmd; }
    cmd *operator->() { return &m_cmd; }
    const cmd &operator*() const { return m_cmd; }
    cmd &operator*() { return m_cmd; }

  private:
    cmd_ptr(cmd cmd) : m_cmd(cmd) {}
    cmd m_cmd;
  };

  struct const_iterator {
  public:
    friend CmdOpSchedule;

    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = cmd;
    using pointer = cmd_ptr;
    using reference = cmd;

    reference operator*() const { return require_cmd(); }
    pointer operator->() const { return cmd_ptr(require_cmd()); }
    reference operator[](difference_type rhs) const {
      const_iterator temp = *this;
      temp += rhs;
      return *temp;
    }

    const_iterator &operator++() {
      m_opIndex += 1;
      m_cmd.m_cmd = nullptr;
      return *this;
    }
    const_iterator operator++(int) {
      const_iterator tmp{*this};
      ++(*this);
      return tmp;
    }

    const_iterator &operator--() {
      m_opIndex -= 1;
      m_cmd.m_cmd = nullptr;
      return *this;
    }
    const_iterator operator--(int) {
      const_iterator tmp{*this};
      ++(*this);
      return tmp;
    }

    inline difference_type operator-(const const_iterator &rhs) const {
      return m_opIndex - rhs.m_opIndex;
    }
    inline const_iterator operator+(const difference_type &rhs) const {
      return const_iterator{m_buffer, m_opIndex + rhs};
    }

    inline const_iterator operator-(const difference_type &rhs) const {
      return const_iterator{m_buffer, m_opIndex - rhs};
    }

    friend inline const_iterator operator+(const difference_type &lhs,
                                           const const_iterator &rhs) {
      return const_iterator(rhs.m_buffer, lhs + rhs.m_opIndex);
    }

    friend inline const_iterator operator-(const difference_type &lhs,
                                           const const_iterator &rhs) {
      return const_iterator(rhs.m_buffer, lhs - rhs.m_opIndex);
    }
    inline const_iterator &operator+=(difference_type rhs) {
      m_opIndex += rhs;
      return *this;
    }
    inline const_iterator &operator-=(difference_type rhs) {
      m_opIndex -= rhs;
      return *this;
    }

    inline bool operator==(const const_iterator &rhs) const {
      return m_opIndex == rhs.m_opIndex;
    }
    inline bool operator!=(const const_iterator &rhs) const {
      return m_opIndex != rhs.m_opIndex;
    }
    inline bool operator>(const const_iterator &rhs) const {
      return m_opIndex > rhs.m_opIndex;
    }
    inline bool operator<(const const_iterator &rhs) const {
      return m_opIndex < rhs.m_opIndex;
    }
    inline bool operator>=(const const_iterator &rhs) const {
      return m_opIndex >= rhs.m_opIndex;
    }
    inline bool operator<=(const const_iterator &rhs) const {
      return m_opIndex <= rhs.m_opIndex;
    }

  private:
    const_iterator(const void *buffer, std::size_t op)
        : m_buffer(buffer), m_opIndex(op), m_cmd(CmdOp::NoOp, nullptr) {}

    cmd require_cmd() const {
      if (m_cmd.m_cmd == nullptr) {
        auto header = getHeader(m_buffer);
        m_cmd.m_op = *reinterpret_cast<const CmdOp *>(
            header->op_offset + m_opIndex * sizeof(CmdOp));
        m_cmd.m_cmd = getCmdBuf(m_buffer, m_opIndex);
      }
      return m_cmd;
    }

    const void *m_buffer;
    std::size_t m_opIndex;

    mutable cmd m_cmd; // just a scatch space.
  };

  const_iterator begin() const { return const_iterator(m_buffer, 0); }

  const_iterator end() const {
    auto header = getHeader(m_buffer);
    return const_iterator(m_buffer, header->op_count);
  }

  const_iterator cbegin() const { return begin(); }
  const_iterator cend() const { return end(); }

  cmd operator[](std::size_t n) const { return begin()[n]; }

  std::size_t size() const {
    auto header = getHeader(m_buffer);
    return header->op_count;
  }

private:
  // NOTE: a noop!
  static inline const CmdOpScheduleHeader *getHeader(const void *buffer) {
    auto oraw = static_cast<const std::byte *>(buffer);
    auto oheader = reinterpret_cast<const CmdOpScheduleHeader *>(oraw);
    return oheader;
  }

  static inline std::span<const std::uint64_t> getCmdOT(const void *buffer) {
    auto header = getHeader(buffer);
    auto otptr = static_cast<const std::byte *>(buffer) + header->cmdOT_offset;
    return std::span{reinterpret_cast<const std::uint64_t *>(otptr),
                     header->op_count};
  }
  static inline const void *getCmdBuf(const void *buffer, std::size_t index) {
    auto header = getHeader(buffer);
    auto ot = getCmdOT(buffer);
    return static_cast<const std::byte *>(buffer) +
           (header->cmdBuf_offset + ot[index]);
  }

  void *m_buffer;
};

} // namespace vkcnn
