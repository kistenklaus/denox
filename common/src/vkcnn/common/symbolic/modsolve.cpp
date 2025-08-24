#include "./modsolve.hpp"
#include "vkcnn/common/containers/small_vector.hpp"
#include <cassert>
#include <fmt/base.h>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::sym {

using value_type = std::uint64_t;
using coef_type = std::int64_t;
using symbol = std::uint64_t;

struct SymbolOrConstant {
  union uni {
    struct bitfield {
      symbol isConstant : 1;
      symbol opt : 1;
      symbol value : 62;
    } bitfield;
    symbol sym;
  } u;
};

struct BinaryExprSymbol {
  SymbolOrConstant a;
  SymbolOrConstant b;
};

struct BinaryExprSymbolHash {
  size_t operator()(const BinaryExprSymbol &k) const noexcept {
    std::hash<symbol> hasher;
    std::size_t h1 = hasher(k.a.u.sym);
    std::size_t h2 = hasher(k.b.u.sym);
    return h1 ^ (h2 << 1);
  }
};

struct BinaryExprSymbolKeyEqual {

  bool operator()(const BinaryExprSymbol &a,
                  const BinaryExprSymbol &b) const noexcept {
    return a.a.u.sym == b.a.u.sym && a.b.u.sym == b.b.u.sym;
  }
};

using BinarySymbolMap =
    std::unordered_map<BinaryExprSymbol, symbol, BinaryExprSymbolHash,
                       BinaryExprSymbolKeyEqual>;

struct AffineCoef {
  symbol sym;
  coef_type factor;
};

struct AffineExpr {
  containers::small_vector<AffineCoef, 2> coef;
  coef_type constant = 0;
};

struct ModExpr {
  AffineExpr affine;
  std::optional<value_type> residue;
};

struct ModSolverState {
  std::vector<ModExpr> facts;
  BinarySymbolMap mulSymbols;
  BinarySymbolMap divSymbols;
  symbol nextSymbol;
};

static void affine_add_sym(AffineExpr &lhs, symbol s, coef_type factor) {
  if (factor == 0) {
    return;
  }
  auto it = lhs.coef.begin();
  while (it != lhs.coef.end()) {
    if (it->sym == s) {
      it->factor += factor;
      if (it->factor == 0) {
        lhs.coef.erase(it);
      }
      return;
    }
    ++it;
  }
  lhs.coef.emplace_back(s, factor);
}

static AffineExpr affine_add(const AffineExpr &lhs, const AffineExpr &rhs) {
  AffineExpr out;
  out.constant = lhs.constant + rhs.constant;
  out.coef = lhs.coef;
  for (std::size_t bx = 0; bx < rhs.coef.size(); ++bx) {
    const auto &coef = rhs.coef[bx];
    affine_add_sym(out, coef.sym, coef.factor);
  }
  return out;
}

static AffineExpr affine_sub(const AffineExpr &lhs, const AffineExpr &rhs) {
  AffineExpr out;
  out.constant = lhs.constant - rhs.constant;
  out.coef = lhs.coef;
  for (std::size_t bx = 0; bx < rhs.coef.size(); ++bx) {
    const auto &coef = rhs.coef[bx];
    affine_add_sym(out, coef.sym, -coef.factor);
  }
  return out;
}

static symbol require_binary_mul_symbol(symbol a, symbol b,
                                        ModSolverState &state) {
  BinaryExprSymbol key{SymbolOrConstant{.u = {.sym = a}},
                       SymbolOrConstant{.u = {.sym = b}}};
  auto it = state.mulSymbols.find(key);
  if (it == state.mulSymbols.end()) {
    symbol s = state.nextSymbol++;
    state.mulSymbols.insert(it, std::make_pair(key, s));
    fmt::println("New Symbol {} : {} * {}", s, a, b);
    return s;
  } else {
    return it->second;
  }
}

static AffineExpr affine_mul(const AffineExpr &lhs, const coef_type &rhs) {
  AffineExpr out = lhs;
  for (std::size_t c = 0; c < out.coef.size(); ++c) {
    out.coef[c].factor *= rhs;
  }
  out.constant *= rhs;
  return out;
}

static AffineExpr affine_mul(const AffineExpr &lhs, const AffineExpr &rhs,
                             ModSolverState &state) {
  if (lhs.coef.empty()) {
    return affine_mul(rhs, lhs.constant);
  }
  if (rhs.coef.empty()) {
    return affine_mul(lhs, rhs.constant);
  }

  AffineExpr out;
  // Constant term.
  out.constant = lhs.constant * rhs.constant;

  // Linear tems scaled by the other side's constant.
  if (lhs.constant != 0) {
    for (const auto &coef : rhs.coef) {
      affine_add_sym(out, coef.sym, coef.factor * lhs.constant);
    }
  }
  if (rhs.constant != 0) {
    for (const auto &coef : lhs.coef) {
      affine_add_sym(out, coef.sym, coef.factor * rhs.constant);
    }
  }
  for (const auto ac : lhs.coef) {
    for (const auto bc : rhs.coef) {
      const symbol s = require_binary_mul_symbol(ac.sym, bc.sym, state);
      affine_add_sym(out, s, ac.factor * bc.factor);
    }
  }
  return out;
}

static symbol require_binary_div_symbol(SymbolOrConstant a, SymbolOrConstant b,
                                        ModSolverState &state) {
  BinaryExprSymbol key{a, b};
  auto it = state.divSymbols.find(key);
  if (it == state.divSymbols.end()) {
    symbol s = state.nextSymbol++;
    state.divSymbols.insert(it, std::make_pair(key, s));
    return s;
  } else {
    return it->second;
  }
}

// Returns true iff A == k * B (all coeffs + constant), and writes k_out.
// No allocations; quadratic scan is fine because coef lists are tiny.
static bool affine_scalar_multiple(const AffineExpr &a, const AffineExpr &b,
                                   coef_type &k_out) {
  bool similar = a.coef.size() == b.coef.size();
  for (std::size_t ax = 0; ax < a.coef.size() && similar; ++ax) {
    bool exists = false;
    for (std::size_t bx = 0; bx < b.coef.size(); ++bx) {
      if (a.coef[ax].sym == b.coef[bx].sym) {
        exists = true;
        break;
      }
    }
    if (!exists) {
      similar = false;
    }
  }
  if (!similar) {
    return false;
  }
  assert(a.coef.size() == b.coef.size());
  // Denominator can't be the zero expression.
  if (b.coef.empty() && b.constant == 0)
    return false;

  bool have_k = false;
  coef_type k = 0;

  // If B has a constant, fix k from constants.
  if (b.constant != 0) {
    if (a.constant % b.constant != 0)
      return false;
    k = a.constant / b.constant;
    have_k = true;
  } else {
    // Then A's constant must be 0 as well.
    if (a.constant != 0)
      return false;
  }

  // For each term in B, find matching term in A and check the ratio.
  std::size_t matched = 0;
  for (const auto &bt : b.coef) {
    // Find bt.sym in A
    const AffineCoef *atp = nullptr;
    for (const auto &at : a.coef) {
      if (at.sym == bt.sym) {
        atp = &at;
        break;
      }
    }
    if (!atp)
      return false; // missing symbol

    // Determine or verify k
    if (!have_k) {
      if (bt.factor == 0)
        return false; // shouldn't happen in canonical form
      if (atp->factor % bt.factor != 0)
        return false;
      k = atp->factor / bt.factor;
      have_k = true;
    } else {
      if (atp->factor != k * bt.factor)
        return false;
    }
    ++matched;
  }

  // Ensure A has no extra symbols beyond B
  if (matched != a.coef.size())
    return false;

  k_out = k;
  return true;
}

static AffineExpr affine_div(const AffineExpr &lhs, symbol lhsid,
                             const coef_type rhs, ModSolverState &state,
                             bool ceilMode) {
  // NOTE: Required because in bitfields unsigned underflow is undefined!
  assert(rhs > 0);
  if (rhs == 1) {
    return lhs;
  }
  if (lhs.coef.empty()) {
    // NOTE: Constant folding.
    assert(lhs.constant > 0);
    AffineExpr out;
    if (ceilMode) {
      out.constant = (lhs.constant + rhs - 1) / rhs;
    } else {
      out.constant = lhs.constant / rhs;
    }
    return out;
  }

  // Check if lhs.constant and all lhs.coef are divisibly be rhs.constant
  coef_type div = rhs;
  assert(div != 0);
  bool allDivisible = true;
  if (lhs.constant % div != 0) {
    // bail.
    allDivisible = false;
  }
  for (std::size_t c = 0; c < lhs.coef.size() && allDivisible; ++c) {
    if (lhs.coef[c].factor % div != 0) {
      allDivisible = false;
    }
  }
  AffineExpr out;
  if (allDivisible) {
    out = lhs;
    for (auto &coef : out.coef) {
      coef.factor /= div;
    }
    out.constant /= div;
  } else {
    SymbolOrConstant a{.u = {.bitfield = {.isConstant = false,
                                          .opt = ceilMode,
                                          .value = lhsid}}};
    SymbolOrConstant b{.u = {.bitfield = {.isConstant = true,
                                          .opt = ceilMode,
                                          .value = static_cast<symbol>(rhs)}}};
    symbol s = require_binary_div_symbol(a, b, state);
    out.coef.emplace_back(s, 1);
    out.constant = 0;
  }
  return out;
}

static AffineExpr affine_div(const AffineExpr &lhs, symbol lhsid,
                             const AffineExpr &rhs, symbol rhsid,
                             ModSolverState &state, bool ceilMode) {
  if (rhs.coef.empty()) {
    assert(rhs.constant > 0);
    return affine_div(lhs, lhsid, rhs.constant, state, ceilMode);
  }
  SymbolOrConstant a{
      .u = {.bitfield = {.isConstant = lhs.coef.empty(),
                         .opt = ceilMode,
                         .value = lhs.coef.empty()
                                      ? static_cast<symbol>(lhs.constant)
                                      : lhsid}}};
  SymbolOrConstant b{
      .u = {.bitfield = {.isConstant = rhs.coef.empty(),
                         .opt = ceilMode,
                         .value = rhs.coef.empty()
                                      ? static_cast<symbol>(rhs.constant)
                                      : rhsid}}};

  AffineExpr out;
  if (lhs.coef.empty()) {
    assert(!rhs.coef.empty());
    // bail. (constant divided by non-constant)
    symbol s = require_binary_div_symbol(a, b, state);
    out.coef.emplace_back(s, 1);
    return out;
  }

  if (lhs.coef.size() == 1 && rhs.coef.size() == 1 &&
      lhs.coef.front().sym == rhs.coef.front().sym && lhs.constant == 0 &&
      rhs.constant == 0) {
    // Single symbol cancels single symbol.
    auto a = lhs.coef.front();
    auto b = rhs.coef.front();
    assert(a.sym == b.sym);
    coef_type con;
    if (ceilMode) {
      con = (a.factor + b.factor - 1) / b.factor;
    } else {
      con = a.factor / b.factor;
    }
    out.constant = con;
    return out;
  }

  // check for scalar multiple (exact match lhs*k / rhs = k)
  coef_type k;
  if (affine_scalar_multiple(lhs, rhs, k)) {
    out.constant = k;
    return out;
  }

  symbol s = require_binary_div_symbol(a, b, state);
  out.coef.emplace_back(s, 1);
  return out;
}

std::optional<std::uint64_t>
modsolve(const hypergraph::ConstGraph<SymValue, SymExpr> &graph,
         hypergraph::NodeId target, std::uint64_t mod) {
  ModSolverState state;
  state.facts.resize(target + 1);
  state.nextSymbol = state.facts.size();

  // Iterate in topological order!
  // NOTE: This assumes that the nodes in a SymAdjGraph are topologically
  // ordered an monotone, which is ensured during construction.
  for (symbol sym = 0; sym <= target; ++sym) {
    hypergraph::NodeId nid{sym};
    fmt::println("Visit {}", sym);
    const auto &n = graph.get(nid);
    if (n.isConstant()) {
      value_type constant = n.value();
      state.facts[sym].affine.constant = constant;
      state.facts[sym].residue = constant % mod;
      continue;
    }

    auto incoming = graph.incoming(nid);
    if (incoming.size() == 0) {
      // NOTE: nid is a parameter.
      state.facts[sym].affine.coef.emplace_back(nid, 1);
      continue;
    }
    // NOTE: All nodes are only ever produced by a single expr (edge)
    assert(incoming.size() == 1);
    const auto eid = incoming[0];
    const auto &expr = graph.get(eid);
    const auto srcs = graph.src(eid);
    // NOTE: All expresisons are binary expressions.
    assert(srcs.size() == 2);
    const auto &aid = static_cast<symbol>(srcs[0]);
    const auto &bid = static_cast<symbol>(srcs[1]);

    const auto &afact = state.facts[aid];
    const auto &bfact = state.facts[bid];

    // Apply affine expr.
    switch (expr) {
    case SymExpr::CeilDiv:
      state.facts[sym].affine =
          affine_div(afact.affine, aid, bfact.affine, bid, state, true);
      break;
    case SymExpr::FloorDiv:
      state.facts[sym].affine =
          affine_div(afact.affine, aid, bfact.affine, bid, state, false);
      break;
    case SymExpr::AlignUp:
      break;
    case SymExpr::Mod:
      break;
    case SymExpr::Sub:
      state.facts[sym].affine = affine_sub(afact.affine, bfact.affine);
      break;
    case SymExpr::Mul:
      fmt::println("Mul v({}) * v({})", aid, bid);
      state.facts[sym].affine = affine_mul(afact.affine, bfact.affine, state);
      break;
    case SymExpr::Add:
      state.facts[sym].affine = affine_add(afact.affine, bfact.affine);
      break;
    case SymExpr::Max:
      break;
    case SymExpr::Min:
      break;
    }
  }

  for (std::size_t n = 0; n < state.facts.size(); ++n) {
    fmt::print("v({}) = ", n);
    const auto &affine = state.facts[n].affine;
    for (std::size_t c = 0; c < affine.coef.size(); ++c) {
      if (c != 0) {
        fmt::print(" + ");
      }
      fmt::print("v({}) * {}", static_cast<std::uint64_t>(affine.coef[c].sym),
                 affine.coef[c].factor);
    }
    if (affine.coef.size() == 0) {
      fmt::print("{}", affine.constant);
    } else {
      fmt::print(" + {}", affine.constant);
    }
    fmt::print(" mod {} = ", mod);
    if (state.facts[n].residue.has_value()) {
      fmt::println("{}", *state.facts[n].residue);
    } else {
      fmt::println("?");
    }
  }

  return std::nullopt;
}

} // namespace vkcnn::sym
