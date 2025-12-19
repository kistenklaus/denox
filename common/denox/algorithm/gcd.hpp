#pragma once

#include <numeric>

namespace denox::algorithm {

template <typename M, typename N>
constexpr inline std::common_type_t<M, N> gcd(M m, N n) {
  return std::gcd(m, n);
}

} // namespace denox::algorithms
