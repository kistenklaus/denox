#pragma once

#include <cstddef>
#include <compare>
#include <cstdint>

namespace vkcnn::hypergraph {

struct EdgeId {
public:
  explicit constexpr EdgeId(std::uint64_t id) : m_id(id) {}

  constexpr operator std::uint64_t() const { return m_id; }

  constexpr auto operator<=>(const EdgeId &) const = default;

private:
  std::uint64_t m_id;
};

} // namespace vkcnn::hypergraph
