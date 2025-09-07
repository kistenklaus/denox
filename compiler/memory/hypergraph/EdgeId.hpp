#pragma once

#include <cstdint>

namespace denox::memory {

struct EdgeId {
public:
  explicit constexpr EdgeId(std::uint64_t id) : m_id(id) {}

  constexpr operator std::uint64_t() const { return m_id; }

  friend bool operator==(const EdgeId &lhs, const EdgeId &rhs) {
    return lhs.m_id == rhs.m_id;
  }

  friend bool operator!=(const EdgeId &lhs, const EdgeId &rhs) {
    return lhs.m_id != rhs.m_id;
  }

private:
  std::uint64_t m_id;
};

} // namespace vkcnn::hypergraph
