#pragma once

#include <cstdint>
#include <limits>

namespace denox::memory {

struct NodeId {
public:
  static constexpr std::uint64_t NullId{
      std::numeric_limits<std::uint64_t>::max()};

  explicit constexpr NodeId(std::uint64_t id) : m_id(id) {}

  constexpr operator std::uint64_t() const { return m_id; }

  friend bool operator==(const NodeId &lhs, const NodeId &rhs) {
    return lhs.m_id == rhs.m_id;
  }

  friend bool operator!=(const NodeId &lhs, const NodeId &rhs) {
    return lhs.m_id != rhs.m_id;
  }

private:
  std::uint64_t m_id;
};

} // namespace vkcnn::hypergraph
