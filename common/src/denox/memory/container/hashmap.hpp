#pragma once

#include <functional>
#include <unordered_map>
#include <unordered_set>
namespace denox::memory {

template <typename Key, typename Value, typename Hash = std::hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using hash_map = std::unordered_map<Key, Value, Hash, KeyEqual>;

template <typename Key, typename Hash = std::hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using hash_set = std::unordered_set<Key, Hash, KeyEqual>;

} // namespace denox::memory
