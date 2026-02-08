#pragma once
#include "denox/symbolic/SymGraph.hpp"

namespace denox {

template <typename L, typename R>
  requires(std::same_as<L, Sym> || std::is_integral_v<L>) &&
          (std::same_as<R, Sym> || std::is_integral_v<R>)
Sym SymGraph::add(L lhs, R rhs, bool dno) {
  Sym a;
  if constexpr (std::same_as<L, Sym>) {
    a = resolve(lhs);
  } else {
    a = Sym::Const(lhs);
  }
  Sym b;
  if constexpr (std::same_as<R, Sym>) {
    b = resolve(rhs);
  } else {
    b = Sym::Const(rhs);
  }
  return add_xx(a, b, dno);
}

template <typename X, typename Y, typename Z>
  requires(std::same_as<X, Sym> || std::is_integral_v<X>) &&
          (std::same_as<Y, Sym> || std::is_integral_v<Y>) &&
          (std::same_as<Z, Sym> || std::is_integral_v<Z>)
Sym SymGraph::add(X x, Y y, Z z, bool dno) {
  return add(add(x, y, false), z, dno);
}

template <typename X, typename Y, typename Z, typename W>
  requires(std::same_as<X, Sym> || std::is_integral_v<X>) &&
          (std::same_as<Y, Sym> || std::is_integral_v<Y>) &&
          (std::same_as<Z, Sym> || std::is_integral_v<Z>) &&
          (std::same_as<W, Sym> || std::is_integral_v<W>)
Sym SymGraph::add(X x, Y y, Z z, W w, bool dno) {
  return add(add(x, y, false), add(z, w, false), dno);
}

template <typename L, typename R>
  requires(std::same_as<L, Sym> || std::is_integral_v<L>) &&
          (std::same_as<R, Sym> || std::is_integral_v<R>)
Sym SymGraph::sub(L lhs, R rhs, bool dno) {
  Sym a;
  if constexpr (std::same_as<L, Sym>) {
    a = resolve(lhs);
  } else {
    a = Sym::Const(lhs);
  }
  Sym b;
  if constexpr (std::same_as<R, Sym>) {
    b = resolve(rhs);
  } else {
    b = Sym::Const(rhs);
  }
  return sub_xx(a, b, dno);
}

template <typename L, typename R>
  requires(std::same_as<L, Sym> || std::is_integral_v<L>) &&
          (std::same_as<R, Sym> || std::is_integral_v<R>)
Sym SymGraph::mul(L lhs, R rhs, bool dno) {
  Sym a;
  if constexpr (std::same_as<L, Sym>) {
    a = resolve(lhs);
  } else {
    a = Sym::Const(lhs);
  }
  Sym b;
  if constexpr (std::same_as<R, Sym>) {
    b = resolve(rhs);
  } else {
    b = Sym::Const(rhs);
  }
  return mul_xx(a, b, dno);
}

template <typename X, typename Y, typename Z>
  requires(std::same_as<X, Sym> || std::is_integral_v<X>) &&
          (std::same_as<Y, Sym> || std::is_integral_v<Y>) &&
          (std::same_as<Z, Sym> || std::is_integral_v<Z>)
Sym SymGraph::mul(X x, Y y, Z z, bool dno) {
  return mul(mul(x, y, false), z, dno);
}

template <typename X, typename Y, typename Z, typename W>
  requires(std::same_as<X, Sym> || std::is_integral_v<X>) &&
          (std::same_as<Y, Sym> || std::is_integral_v<Y>) &&
          (std::same_as<Z, Sym> || std::is_integral_v<Z>) &&
          (std::same_as<W, Sym> || std::is_integral_v<W>)
Sym SymGraph::mul(X x, Y y, Z z, W w, bool dno) {
  return mul(mul(x, y, false), mul(z, w, false), dno);
}

template <typename L>
  requires(std::same_as<L, Sym> || std::is_integral_v<L>)
Sym SymGraph::pow(L lhs, value_type rhs, bool dno) {
  Sym p = Sym::Const(1);
  if (rhs > 0) {
    for (std::size_t n = 0; n < static_cast<std::size_t>(rhs); ++n) {
      p = mul(p, lhs);
    }
    return p;
  } else if (rhs < 0) {
    // NOTE: 1 / X == 1 / X^2 == 1 / X^3 (unsigned arithmetics)
    return div(1, lhs);
  } else {
    return require_const_sym(1, dno);
  }
}

template <typename L, typename R>
  requires(std::same_as<L, Sym> || std::is_integral_v<L>) &&
          (std::same_as<R, Sym> || std::is_integral_v<R>)
Sym SymGraph::div(L lhs, R rhs, bool dno) {
  Sym a;
  if constexpr (std::same_as<L, Sym>) {
    a = resolve(lhs);
  } else {
    a = Sym::Const(lhs);
  }
  Sym b;
  if constexpr (std::same_as<R, Sym>) {
    b = resolve(rhs);
  } else {
    b = Sym::Const(rhs);
  }
  return div_xx(a, b, dno);
}

template <typename L, typename R>
  requires(std::same_as<L, Sym> || std::is_integral_v<L>) &&
          (std::same_as<R, Sym> || std::is_integral_v<R>)
Sym SymGraph::cdiv(L lhs, R rhs, bool dno) {
  const auto r = resolve(rhs);
  if (r.isConstant() && r.constant() - 1 >= 0) {
    // NOTE: This helps to avoid some of the mod blowup 
    Sym divisibilityCheck = mod(lhs, r.constant());
    if (divisibilityCheck.isConstant() && divisibilityCheck.constant() == 0) {
      return div(lhs, r.constant());
    }
    return div(add(lhs, r.constant() - 1), rhs, dno);
  }
  return div(sub(add(lhs, rhs, false), 1), rhs, dno);
}

template <typename L, typename R>
  requires(std::same_as<L, Sym> || std::is_integral_v<L>) &&
          (std::same_as<R, Sym> || std::is_integral_v<R>)
Sym SymGraph::mod(L lhs, R rhs, bool dno) {
  Sym a;
  if constexpr (std::same_as<L, Sym>) {
    a = resolve(lhs);
  } else {
    a = Sym::Const(lhs);
  }
  Sym b;
  if constexpr (std::same_as<R, Sym>) {
    b = resolve(rhs);
  } else {
    b = Sym::Const(rhs);
  }
  return mod_xx(a, b, dno);
}

template <typename L, typename R>
  requires(std::same_as<L, Sym> || std::is_integral_v<L>) &&
          (std::same_as<R, Sym> || std::is_integral_v<R>)
Sym SymGraph::alignUp(L sym, R alignment, bool dno) {
  return mul(cdiv(sym, alignment, false), alignment, dno);
}

template <typename L, typename R>
  requires(std::same_as<L, Sym> || std::is_integral_v<L>) &&
          (std::same_as<R, Sym> || std::is_integral_v<R>)
Sym SymGraph::alignDown(L sym, R alignment, bool dno) {
  return mul(div(sym, alignment, false), alignment, dno);
}

template <typename E, typename K, typename P, typename S, typename D>
  requires(std::same_as<E, Sym> || std::is_integral_v<E>) &&
          (std::same_as<K, Sym> || std::is_integral_v<K>) &&
          (std::same_as<P, Sym> || std::is_integral_v<P>) &&
          (std::same_as<S, Sym> || std::is_integral_v<S>) &&
          (std::same_as<D, Sym> || std::is_integral_v<D>)
Sym SymGraph::pool(E extent, K kernelSize, P padding, S stride, D dilation,
                   bool dno) {
  // (E + 2 * P - ((K - 1) * D +1)) div S + 1
  auto num = sub(add(extent, mul(2, padding, false), false),
                 add(mul(sub(kernelSize, 1, false), dilation, false), 1, false),
                 false);
  auto denom = stride;
  return add(div(num, denom, false), 1, dno);
}

template <typename E, typename K, typename P, typename S, typename D>
  requires(std::same_as<E, Sym> || std::is_integral_v<E>) &&
          (std::same_as<K, Sym> || std::is_integral_v<K>) &&
          (std::same_as<P, Sym> || std::is_integral_v<P>) &&
          (std::same_as<S, Sym> || std::is_integral_v<S>) &&
          (std::same_as<D, Sym> || std::is_integral_v<D>)
Sym SymGraph::cpool(E extent, K kernelSize, P padding, S stride, D dilation,
                    bool dno) {
  // (E + 2 * P - ((K - 1) * D +1)) cdiv S + 1
  auto num = sub(add(extent, mul(2, padding)),
                 add(mul(sub(kernelSize, 1), dilation), 1));
  auto denom = stride;
  return add(cdiv(num, denom), 1, dno);
}

template <typename L, typename R>
  requires(std::same_as<L, Sym> || std::is_integral_v<L>) &&
          (std::same_as<R, Sym> || std::is_integral_v<R>)
Sym SymGraph::min(L lhs, R rhs, bool dno) {
  Sym a;
  if constexpr (std::same_as<L, Sym>) {
    a = resolve(lhs);
  } else {
    a = Sym::Const(lhs);
  }
  Sym b;
  if constexpr (std::same_as<R, Sym>) {
    b = resolve(rhs);
  } else {
    b = Sym::Const(rhs);
  }
  return min_xx(a, b, dno);
}

template <typename L, typename R>
  requires(std::same_as<L, Sym> || std::is_integral_v<L>) &&
          (std::same_as<R, Sym> || std::is_integral_v<R>)
Sym SymGraph::max(L lhs, R rhs, bool dno) {
  Sym a;
  if constexpr (std::same_as<L, Sym>) {
    a = resolve(lhs);
  } else {
    a = Sym::Const(lhs);
  }
  Sym b;
  if constexpr (std::same_as<R, Sym>) {
    b = resolve(rhs);
  } else {
    b = Sym::Const(rhs);
  }
  return max_xx(a, b, dno);
}

} // namespace denox
