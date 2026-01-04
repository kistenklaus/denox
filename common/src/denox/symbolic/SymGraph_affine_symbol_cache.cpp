#include "denox/symbolic/SymGraph.hpp"
#include "denox/memory/container/optional.hpp"

namespace denox {

SymGraph::symbol SymGraph::next_sym() {
  Expr expr;
  symbol s = m_expressions.size();
  m_expressions.emplace_back();
  return s;
}
SymGraph::symbol SymGraph::create_variable(ExprType type) {
  assert(type == ExprType::Identity || type == ExprType::NonAffine);
  symbol s = next_sym();
  m_expressions[s].expr = type;
  m_expressions[s].affine.constant = 0;
  m_expressions[s].affine.coef.emplace_back(s, 1);
  m_affineCache.insert(std::make_pair(m_expressions[s].affine, s));
  return s;
}

Sym SymGraph::require_affine_sym(ExprType type, Sym lhs, Sym rhs,
                                 const AffineExpr &affine, bool dno) {
  assert(type != ExprType::NonAffine);
  if (!dno && affine.isPureConstant()) {
    return Sym::Const(affine.constant);
  }

  auto it = m_affineCache.find(affine);
  if (it == m_affineCache.end()) {
    symbol s = next_sym();
    m_expressions[s].expr = type;
    m_expressions[s].affine = affine;
    m_expressions[s].lhs = lhs;
    m_expressions[s].rhs = rhs;
    m_affineCache.insert(it, std::make_pair(affine, s));
    return Sym::Symbol(s);
  } else {
    return Sym::Symbol(it->second);
  }
}

Sym SymGraph::require_const_sym(value_type constant, bool dno) {
  AffineExpr affine;
  affine.constant = constant;
  return require_affine_sym(ExprType::Const, Sym::Const(constant),
                            Sym::Const(0), affine, dno);
}

denox::memory::optional<SymGraph::symbol>
SymGraph::find_sym_of_affine(const AffineExpr &expr) {
  auto it = m_affineCache.find(expr);
  if (it == m_affineCache.end()) {
    return denox::memory::nullopt;
  } else {
    return it->second;
  }
}

denox::memory::optional<Sym>
SymGraph::try_construct_affine_sym(const AffineExpr &expr, symbol hint,
                                   std::size_t depth, bool dno) {
  {
    // NOTE: This should catch : hint == expr, does a lookup in a cache, but
    // can fail and will never create new symbols.
    if (auto symopt = find_sym_of_affine(expr)) {
      return Sym::Symbol(*symopt);
    }
  }
  if (depth == 0) {
    return denox::memory::nullopt;
  }
  auto s = m_expressions[hint];
  const AffineExpr &similar = s.affine;
  {
    AffineExpr delta = affine_sub(expr, similar);
    if (delta.isPureConstant()) {
      if (delta.constant > 0) {
        return add_sc(hint, delta.constant, dno);
      } else if (delta.constant < 0) {
        return sub_sc(hint, -delta.constant, dno);
      } else {
        return Sym::Symbol(hint);
      }
    }
    if (delta.constant == 0 && delta.coef.size() == 1) {
      auto coef = delta.coef[0];
      if (coef.factor == 1) {
        return add_ss(coef.sym, hint, dno);
      } else if (coef.factor == -1) {
        return sub_ss(hint, coef.sym, dno);
      }
    }
  }
  if (s.lhs.isSymbolic()) {
    const auto &next = m_expressions[s.lhs.sym()];
    if (next.expr != ExprType::Identity && next.expr != ExprType::NonAffine) {
      denox::memory::optional<Sym> lhsAttempt =
          try_construct_affine_sym(expr, s.lhs.sym(), depth - 1, dno);
      if (lhsAttempt.has_value()) {
        return *lhsAttempt;
      }
    }
  }
  if (s.rhs.isSymbolic()) {
    const auto &next = m_expressions[s.rhs.sym()];
    if (next.expr != ExprType::Identity && next.expr != ExprType::NonAffine) {
      denox::memory::optional<Sym> rhsAttempt =
          try_construct_affine_sym(expr, s.rhs.sym(), depth - 1, dno);
      if (rhsAttempt.has_value()) {
        return *rhsAttempt;
      }
    }
  }
  return denox::memory::nullopt;
}
Sym SymGraph::construct_affine_sym(AffineExpr expr, bool dno,
                                   denox::memory::optional<symbol> hint) {
  if (expr.coef.size() == 1 && expr.constant == 0 && expr.coef[0].factor == 1) {
    return require_affine_sym(ExprType::Mul, Sym::Symbol(expr.coef[0].sym),
                              Sym::Const(expr.coef[0].factor), expr, dno);
  }
  if (expr.isPureConstant()) {
    return require_const_sym(expr.constant, dno);
  }

  if (auto opt = find_sym_of_affine(expr)) {
    return Sym::Symbol(*opt);
  }

  if (hint.has_value()) {
    static constexpr std::size_t SEARCH_DEPTH = 3;
    if (auto opt = try_construct_affine_sym(expr, *hint, SEARCH_DEPTH, dno)) {
      if (opt->isSymbolic()) {
        assert(symbolic::details::AffineExprComp{}(
            m_expressions[opt->sym()].affine, expr));
      } else {
        assert(expr.constant == opt->constant());
      }
      return *opt;
    }
  }

  Sym acc = Sym::Const(expr.constant);
  for (const auto &c : expr.coef) {
    acc = add_xx(acc, mul_sc(c.sym, c.factor, false), false);
  }

  if (acc.isConstant()) {
    return require_const_sym(acc.constant(), dno);
  } else {
    return mul_sc(acc.sym(), 1, dno); // retain constants.
  }
}

} // namespace denox::compiler
