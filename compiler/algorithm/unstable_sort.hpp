#pragma once

#include <algorithm>
#include <functional>
#include <iterator>

namespace denox::algorithm {

template <typename RandomIt,
          typename Compare =
              std::less<typename std::iterator_traits<RandomIt>::value_type>>
void unstable_sort(RandomIt first, RandomIt last, Compare compare = {}) {
  return std::sort(first, last, compare);
}

} // namespace denox::algorithms
