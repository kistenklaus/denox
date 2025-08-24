#pragma once

#include <cstddef>
#include <compare>

namespace vkcnn::hypergraph {

struct EdgeId {
public:
  explicit constexpr EdgeId(std::size_t id) : m_id(id) {}

  constexpr operator std::size_t() const { return m_id; }

  constexpr auto operator<=>(const EdgeId &) const = default;

private:
  std::size_t m_id;
};

} // namespace vkcnn::hypergraph
