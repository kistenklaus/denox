#pragma once
#include "denox/memory/container/dynamic_bitset.hpp"
#include "denox/memory/container/vector.hpp"
#include <cstddef>
#include <numeric>
#include <ranges>

namespace denox {

template <typename Container = memory::vector<size_t>>
  requires std::ranges::random_access_range<Container> &&
           std::ranges::sized_range<Container>
struct union_find {
public:
  explicit union_find(size_t n) : m_parent(n), m_rank(n, 0) {
    std::iota(m_parent.begin(), m_parent.end(), size_t{0});
  }

  size_t size() const noexcept { return m_parent.size(); }

  size_t find(size_t x) noexcept {
    size_t root = x;
    while (m_parent[root] != root) {
      root = m_parent[root];
    }

    // Path compression
    while (m_parent[x] != x) {
      size_t next = m_parent[x];
      m_parent[x] = root;
      x = next;
    }

    return root;
  }

  bool unite(size_t a, size_t b) noexcept {
    a = find(a);
    b = find(b);

    if (a == b) {
      return false;
    }

    if (m_rank[a] < m_rank[b]) {
      m_parent[a] = b;
    } else if (m_rank[a] > m_rank[b]) {
      m_parent[b] = a;
    } else {
      m_parent[b] = a;
      ++m_rank[a];
    }
    return true;
  }

  memory::vector<std::size_t> roots() {
    memory::dynamic_bitset seen(size(), false);
    memory::vector<std::size_t> result;
    result.reserve(size());

    for (std::size_t i = 0; i < size(); ++i) {
      std::size_t r = find(i);
      if (!seen[r]) {
        seen[r] = true;
        result.push_back(r);
      }
    }

    return result;
  }

  memory::vector<memory::vector<size_t>> components() {
    memory::vector<size_t> root_index(size(), static_cast<size_t>(-1));
    memory::vector<memory::vector<size_t>> comps;

    for (size_t i = 0; i < size(); ++i) {
      size_t r = find(i);

      if (root_index[r] == static_cast<size_t>(-1)) {
        root_index[r] = comps.size();
        comps.emplace_back();
      }

      comps[root_index[r]].push_back(i);
    }

    return comps;
  }

private:
  Container m_parent;
  Container m_rank;
};

} // namespace denox
