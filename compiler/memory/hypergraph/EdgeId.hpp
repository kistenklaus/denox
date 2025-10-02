#pragma once

#include <cstdint>
#include <limits>

namespace denox::memory {

struct EdgeId {
public:
  explicit EdgeId() : m_id(std::numeric_limits<std::uint64_t>::max()) {}

  explicit constexpr EdgeId(std::uint64_t id) : m_id(id) {}

  constexpr explicit operator std::uint64_t() const { return m_id; }
  constexpr explicit operator bool() const {
    return m_id != std::numeric_limits<std::uint64_t>::max();
  }
  constexpr std::uint64_t operator*() const { return m_id; }

  friend bool operator==(const EdgeId &lhs, const EdgeId &rhs) {
    return lhs.m_id == rhs.m_id;
  }

  friend bool operator!=(const EdgeId &lhs, const EdgeId &rhs) {
    return lhs.m_id != rhs.m_id;
  }

private:
  std::uint64_t m_id;
};

} // namespace denox::memory
