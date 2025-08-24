#pragma once

#include <cstdint>
#include <limits>
#include <compare>

namespace vkcnn::hypergraph {

struct NodeId {
public:
  static constexpr std::uint64_t NullId{
      std::numeric_limits<std::uint64_t>::max()};

  explicit constexpr NodeId(std::uint64_t id) : m_id(id) {}

  constexpr operator std::uint64_t() const { return m_id; }

  constexpr auto operator<=>(const NodeId &) const = default;

private:
  std::uint64_t m_id;
};

} // namespace vkcnn::hypergraph
