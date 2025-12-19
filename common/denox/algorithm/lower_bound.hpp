#pragma once

#include <algorithm>
#include <iterator>

namespace denox::algorithm {

template <typename It,
          typename T = typename std::iterator_traits<It>::value_type,
          typename Compare>
auto lower_bound(It first, It last, T value, Compare comp) {
  return std::lower_bound<It, T, Compare>(first, last, value, comp);
}

} // namespace denox::algorithms
