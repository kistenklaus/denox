#pragma once

#include "denox/memory/container/optional.hpp"
#include "denox/memory/container/vector.hpp"
#include <algorithm>
#include <concepts>

namespace denox::memory {

template <std::equality_comparable K, typename V> class linear_map {
public:
  void insert(const K &key, const V &value) {
    assert(m_keys.size() == m_values.size());
    assert(!containsKey(key));
    m_values.push_back(value);
    m_keys.push_back(key);
  }

  void insert(const K &key, V &&value) {
    assert(m_keys.size() == m_values.size());
    assert(!containsKey(key));
    m_values.push_back(std::move(value));
    m_keys.push_back(key);
  }

  bool erase(const K &key) {
    assert(m_keys.size() == m_values.size());
    auto it = std::ranges::find(m_keys, key);
    if (it == m_keys.end()) {
      return false;
    }
    std::size_t index = std::distance(m_keys.begin(), it);
    std::swap(m_keys[index], m_keys.back());
    std::swap(m_values[index], m_values.back());
    m_keys.pop_back();
    m_values.pop_back();
    return true;
  }

  bool containsKey(const K &key) const {
    auto it = std::ranges::find(m_keys, key);
    return it != m_keys.end();
  }

  denox::memory::optional<V> get(const K &key) {
    auto it = std::ranges::find(m_keys, key);
    bool exits = it != m_keys.end();
    if (exits) {
      return *it;
    } else {
      return std::nullopt;
    }
  }

  auto begin() { return m_values.begin(); }
  auto begin() const { return m_values.begin(); }
  auto end() { return m_values.end(); }
  auto end() const { return m_values.end(); }

private:
  denox::memory::vector<K> m_keys;
  denox::memory::vector<V> m_values;
};
} // namespace denox::memory
