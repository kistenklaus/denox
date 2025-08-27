#pragma once
#include "./SymGraph.hpp"
#include <numeric>

namespace vkcnn {

void SymGraph::affine_add_sym(AffineExpr &lhs, symbol s, value_type factor) {
  if (factor == 0)
    return;

  auto &v = lhs.coef;

  auto it =
      std::lower_bound(v.begin(), v.end(), s,
                       [](const AffineCoef &a, symbol b) { return a.sym < b; });

  if (it != v.end() && it->sym == s) {
    const value_type new_factor = it->factor + factor;
    if (new_factor == 0) {
      v.erase(it);
    } else {
      it->factor = new_factor;
    }
    return;
  }

  v.insert(it, AffineCoef{s, factor});
}

SymGraph::AffineExpr SymGraph::affine_add(const AffineExpr &lhs,
                                          const AffineExpr &rhs) {
  AffineExpr out = lhs;
  affine_add_acc(out, rhs);
  return out;
}
void SymGraph::affine_mul_add_acc(AffineExpr &lhs, const AffineExpr &rhs,
                                  value_type factor) {
  lhs.constant += rhs.constant * factor;
  for (std::size_t bx = 0; bx < rhs.coef.size(); ++bx) {
    const auto &coef = rhs.coef[bx];
    affine_add_sym(lhs, coef.sym, coef.factor * factor);
  }
}
void SymGraph::affine_add_acc(AffineExpr &lhs, const AffineExpr &rhs) {
  lhs.constant += rhs.constant;
  for (std::size_t bx = 0; bx < rhs.coef.size(); ++bx) {
    const auto &coef = rhs.coef[bx];
    affine_add_sym(lhs, coef.sym, coef.factor);
  }
}
SymGraph::AffineExpr SymGraph::affine_mul(const AffineExpr &lhs,
                                          const value_type &rhs) {
  AffineExpr out = lhs;
  affine_mul_acc(out, rhs);
  return out;
}
void SymGraph::affine_mul_acc(AffineExpr &lhs, const value_type &rhs) {
  if (rhs == 0) {
    lhs.coef.clear();
    lhs.constant = 0;
    return;
  }
  for (std::size_t c = 0; c < lhs.coef.size(); ++c) {
    lhs.coef[c].factor *= rhs;
  }
  lhs.constant *= rhs;
}

SymGraph::AffineExpr SymGraph::affine_sub(const AffineExpr &lhs,
                                          const AffineExpr &rhs) {
  AffineExpr out = lhs;
  affine_mul_add_acc(out, rhs, -1);
  return out;
}

std::optional<symbolic::details::AffineExpr>
SymGraph::affine_mul(const AffineExpr &lhs, const AffineExpr &rhs) {
  // NOTE: Only affine is one side is constant,
  // otherwise we require terms like X*Y or X^2, which are nonaffine.
  if (lhs.coef.empty()) {
    return affine_mul(rhs, lhs.constant);
  }
  if (rhs.coef.empty()) {
    return affine_mul(lhs, rhs.constant);
  }
  return std::nullopt;
}

std::optional<SymGraph::AffineExpr> SymGraph::affine_div(const AffineExpr &lhs,
                                                         value_type rhs) {
  assert(rhs > 0);
  if (rhs == 1)
    return lhs;

  // All variable coefficients must be divisible by rhs
  AffineExpr out;
  out.coef.reserve(lhs.coef.size());
  for (auto const &c : lhs.coef) {
    if (c.factor % rhs != 0) {
      return std::nullopt; // non-affine residual -> bail
    }
    out.coef.push_back({c.sym, c.factor / rhs});
  }

  // Constant term uses floor division
  auto [q, _] = floordivmod(lhs.constant, rhs);
  out.constant = q;

  return out;
}
std::optional<SymGraph::AffineExpr>
SymGraph::affine_div(const AffineExpr &lhs, const AffineExpr &rhs) {
  if (rhs.coef.empty()) {
    assert(rhs.constant > 0);
    return affine_div(lhs, rhs.constant);
  }

  if (lhs.coef.empty()) {
    if (lhs.constant == 0) {
      AffineExpr affine;
      affine.constant = 0;
      return affine;
    }
    return std::nullopt;
  }

  if (lhs.coef.size() == rhs.coef.size()) {
    bool equalSymbols = true;
    for (std::size_t c = 0; c < lhs.coef.size() && equalSymbols; ++c) {
      if (lhs.coef[c].sym != rhs.coef[c].sym) {
        equalSymbols = false;
      }
    }
    if (equalSymbols) {

      // NOTE: A / B = (k(A')+r) div h(B'), where
      // k = gcd(A-coefficients)
      // A' = A / k
      // r = A-const mod k
      // h = gcd(B)
      // B' = B / h
      //
      // If A' == B', then this collapses into:
      // k(A') div hB' + r div hB'
      value_type k = 0;
      for (const auto &coef : lhs.coef) {
        k = std::gcd(k, std::abs(coef.factor));
      }
      if (lhs.coef.empty()) {
        k = 1;
      }
      AffineExpr A = lhs;
      for (auto &coef : A.coef) {
        coef.factor /= k;
      }
      A.constant = lhs.constant / k;
      value_type r = lhs.constant % k;
      // if (r < 0) {
      //   r += k;
      // }

      value_type h = rhs.constant;
      for (const auto &coef : rhs.coef) {
        h = std::gcd(h, std::abs(coef.factor));
      }
      AffineExpr B = rhs;
      for (auto &coef : B.coef) {
        coef.factor /= h;
      }
      B.constant /= h;

      bool AeqB = A.constant == B.constant;
      for (std::size_t c = 0; c < A.coef.size() && AeqB; ++c) {
        if (A.coef[c].factor != B.coef[c].factor) {
          AeqB = false;
        }
      }
      if (AeqB) {
        value_type kh = k / h;

        if (r == 0) {
          AffineExpr affine;
          affine.constant = kh;
          return affine;
        } else if (r < 0 && -r <= h) {
          AffineExpr affine;
          affine.constant = kh - 1;
          return affine;
        } else if (r > 0 && r < h) {
          AffineExpr affine;
          affine.constant = kh;
          return affine;
        }
      }
    }
  }

  if (lhs.coef.size() == 1 && rhs.coef.size() == 1 &&
      lhs.coef.front().sym == rhs.coef.front().sym &&
      lhs.constant == rhs.constant) {
    // Single symbol cancels single symbol
    AffineExpr out;
    auto a = lhs.coef.front();
    auto b = rhs.coef.front();
    assert(a.sym == b.sym);
    out.constant = a.factor / b.factor;
    return out;
  }

  return std::nullopt;
}
std::optional<SymGraph::value_type> SymGraph::affine_mod(const AffineExpr &lhs,
                                               const value_type &rhs) {
  assert(rhs > 0);
  if (rhs == 1) {
    return 0;
  }
  if (lhs.isPureConstant()) {
    assert(lhs.constant >= 0);
    AffineExpr affine;
    return emod(lhs.constant, rhs);
  }

  bool allDiv = true;
  for (auto &coef : lhs.coef) {
    if (coef.factor % rhs != 0) {
      allDiv = false;
      break;
    }
  }
  if (allDiv) {
    return emod(lhs.constant, rhs);
  }
  return std::nullopt;
}
std::optional<SymGraph::AffineExpr> SymGraph::affine_mod(const AffineExpr &lhs,
                                               const AffineExpr &rhs) {
  if (lhs.isPureConstant()) {
    assert(lhs.constant >= 0);
    if (rhs.isPureConstant()) {
      assert(rhs.constant > 0);
      AffineExpr affine;
      affine.constant = lhs.constant % rhs.constant;
      if (affine.constant < 0) {
        affine.constant += rhs.constant;
      }
      return affine;
    } else {
      if (lhs.constant == 0) {
        AffineExpr affine;
        affine.constant = 0;
        return affine;
      }
      return std::nullopt;
    }
  } else {
    if (rhs.isPureConstant()) {
      assert(rhs.constant > 0);
      if (auto mod = affine_mod(lhs, rhs.constant)) {
        AffineExpr affine;
        affine.constant = *mod;
        return affine;
      } else {
        return std::nullopt;
      }
    } else {
      assert(!lhs.isPureConstant() && !rhs.isPureConstant());
      const static auto full_gcd = [](const AffineExpr &e) -> value_type {
        value_type g = 0;
        for (auto &t : e.coef)
          g = std::gcd(g, static_cast<value_type>(std::abs(t.factor)));
        g = std::gcd(g, static_cast<value_type>(std::abs(e.constant)));
        return g == 0 ? 1 : g; // avoid 0
      };
      const static auto divide_all = [](AffineExpr e, value_type k) {
        for (auto &t : e.coef)
          t.factor /= static_cast<value_type>(k);
        e.constant /= static_cast<value_type>(k);
        return e;
      };

      const static auto affine_equal = [](const AffineExpr &a,
                                          const AffineExpr &b) {
        if (a.constant != b.constant || a.coef.size() != b.coef.size())
          return false;
        for (size_t i = 0; i < a.coef.size(); ++i)
          if (a.coef[i].sym != b.coef[i].sym ||
              a.coef[i].factor != b.coef[i].factor)
            return false;
        return true;
      };
      {
        value_type gL = full_gcd(lhs);
        value_type gR = full_gcd(rhs);
        AffineExpr EL = divide_all(lhs, gL);
        AffineExpr ER = divide_all(rhs, gR);
        if (affine_equal(EL, ER)) {
          value_type s = gL % gR;
          if (s < 0) {
            s += gR;
          }
          AffineExpr out;
          for (auto &t : EL.coef) {
            value_type v = t.factor * static_cast<value_type>(s);
            if (v != 0) {
              out.coef.emplace_back(t.sym, v);
            }
          }
          out.constant = EL.constant * static_cast<value_type>(s);
          return out;
        }
      }
    }
  }
  return std::nullopt;
}

} // namespace vkcnn
