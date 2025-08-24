#pragma once

#include "vkcnn/common/cmd/CmdOpScheduleHeader.hpp"
#include "vkcnn/common/cmd/record/SizedRecordHeader.hpp"
#include "vkcnn/common/memory/align.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
namespace vkcnn {

class CmdOpDispatch;

class ParameterList {
public:
  friend CmdOpDispatch;

  ParameterList() : m_size(0), m_offsets(nullptr), m_buffer(nullptr) {}

  std::size_t size() const { return m_size; }
  std::span<const std::byte> operator[](std::size_t x) const {
    assert(x < m_size);
    std::uint64_t offset = m_offsets[x];
    auto header = static_cast<const CmdOpScheduleHeader *>(m_buffer);
    auto paramHeaderPtr = header->param_offset +
                          static_cast<const std::byte *>(m_buffer) + offset;
    auto sizedHeader =
        reinterpret_cast<const SizedRecordHeader *>(paramHeaderPtr);
    std::size_t byteSize = sizedHeader->size;
    auto paramPtr = align_up(paramHeaderPtr, alignof(double));
    return std::span{paramPtr, byteSize};
  }

  struct const_iterator {
    friend ParameterList;

    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = std::span<const std::byte>;
    using pointer = std::span<const std::byte>;
    using reference = std::span<const std::byte>;

    reference operator*() const { return deref(); }
    pointer operator->() const { return deref(); }
    reference operator[](difference_type rhs) const {
      const_iterator temp = *this;
      temp += rhs;
      return *temp;
    }

    const_iterator &operator++() {
      ++m_offset;
      return *this;
    }
    const_iterator operator++(int) {
      const_iterator tmp{*this};
      ++(*this);
      return tmp;
    }

    const_iterator &operator--() {
      --m_offset;
      return *this;
    }
    const_iterator operator--(int) {
      const_iterator tmp{*this};
      ++(*this);
      return tmp;
    }

    inline difference_type operator-(const const_iterator &rhs) const {
      return m_offset - rhs.m_offset;
    }
    inline const_iterator operator+(const difference_type &rhs) const {
      return const_iterator{m_buffer, m_offset + rhs};
    }

    inline const_iterator operator-(const difference_type &rhs) const {
      return const_iterator{m_buffer, m_offset - rhs};
    }

    friend inline const_iterator operator+(const difference_type &lhs,
                                           const const_iterator &rhs) {
      return const_iterator(rhs.m_buffer, lhs + rhs.m_offset);
    }

    inline const_iterator &operator+=(difference_type rhs) {
      m_offset += rhs;
      return *this;
    }
    inline const_iterator &operator-=(difference_type rhs) {
      m_offset -= rhs;
      return *this;
    }

    inline bool operator==(const const_iterator &rhs) const {
      return m_offset == rhs.m_offset;
    }
    inline bool operator!=(const const_iterator &rhs) const {
      return m_offset != rhs.m_offset;
    }
    inline bool operator>(const const_iterator &rhs) const {
      return m_offset > rhs.m_offset;
    }
    inline bool operator<(const const_iterator &rhs) const {
      return m_offset < rhs.m_offset;
    }
    inline bool operator>=(const const_iterator &rhs) const {
      return m_offset >= rhs.m_offset;
    }
    inline bool operator<=(const const_iterator &rhs) const {
      return m_offset <= rhs.m_offset;
    }

  private:
    std::span<const std::byte> deref() const {
      std::uint64_t offset = *m_offset;
      auto header = static_cast<const CmdOpScheduleHeader *>(m_buffer);
      auto paramHeaderPtr = header->param_offset +
                            static_cast<const std::byte *>(m_buffer) + offset;
      auto sizedHeader =
          reinterpret_cast<const SizedRecordHeader *>(paramHeaderPtr);
      std::size_t byteSize = sizedHeader->size;
      auto paramPtr = align_up(paramHeaderPtr, alignof(double));
      return std::span{paramPtr, byteSize};
    }
    const_iterator(const void *buffer, const std::uint64_t *offset)
        : m_offset(offset), m_buffer(buffer) {}

    const std::uint64_t *m_offset;
    const void *m_buffer;
  };

  const_iterator begin() const { return const_iterator{m_buffer, m_offsets}; }
  const_iterator cbegin() const { return const_iterator{m_buffer, m_offsets}; }
  const_iterator end() const {
    return const_iterator{m_buffer, m_offsets + m_size};
  }
  const_iterator cend() const { return end(); }

private:
  explicit ParameterList(std::size_t size, const std::uint64_t *offsets,
                         const void *buffer)
      : m_size(size), m_offsets(offsets), m_buffer(buffer) {}
  std::size_t m_size;
  const std::uint64_t *m_offsets;
  const void *m_buffer;
};

} // namespace vkcnn
