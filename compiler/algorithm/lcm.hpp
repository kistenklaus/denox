#pragma once

#include <numeric>

namespace denox::algorithm {

template <typename M, typename N>
constexpr std::common_type_t<M, N> lcm(M m, N n) {
  return std::lcm(m, n);
}

} // namespace denox::algorithms
