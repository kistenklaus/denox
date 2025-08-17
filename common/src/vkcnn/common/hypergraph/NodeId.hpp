#pragma once

#include <cstddef>
namespace vkcnn::hypergraph {

struct NodeId {
public:
  explicit constexpr NodeId(std::size_t id) : m_id(id) {}

  constexpr operator std::size_t() const { return m_id; }

private:
  std::size_t m_id;
};

} // namespace vkcnn::hypergraph
