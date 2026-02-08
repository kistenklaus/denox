#include "denox/algorithm/lcm.hpp"
#include "denox/diag/logging.hpp"
#include "denox/symbolic/SymGraph.hpp"
#include <limits>

namespace denox {

denox::memory::optional<SymGraph::value_type>
SymGraph::modsolve_resume(symbol lhs, Sym m, uint32_t depth) {
  if (m.isSymbolic()) {
    return denox::memory::nullopt;
  }
  if (m.isConstant() && m.constant() >= std::numeric_limits<uint32_t>::max()) {
    return denox::memory::nullopt;
  }
  auto solver = require_modsolver(m);
  return modsolve_resume_solver(solver, lhs, m, depth);
}
memory::optional<SymGraph::ModExpr>
SymGraph::modsolve_reduce_symbol_mod_m(symbol sym, const Sym mod,
                                       uint32_t depth) {
  auto solver = require_modsolver(mod);
  if (solver == nullptr) {
    return memory::nullopt;
  }
  auto r = modsolve_resume_solver(solver, sym, mod, depth);
  if (!r) {
    return memory::nullopt;
  }
  return solver->expressions[sym];
}
denox::memory::optional<SymGraph::ModExpr>
SymGraph::modsolve_reduce_affine_mod_m(const AffineExpr &affine, const Sym msym,
                                       uint32_t depth) {
  auto solver = require_modsolver(msym);
  if (solver == nullptr) {
    return memory::nullopt;
  }

  if (msym.isSymbolic()) {
    return denox::memory::nullopt;
  }
  const auto m = msym.constant();

  ModExpr modexpr;
  modexpr.affine.constant = emod(affine.constant, m);
  for (const auto coef : affine.coef) {
    modsolve_resume_solver(solver, coef.sym, msym, depth);
    const auto &c_affine = solver->expressions[coef.sym].affine;
    modsolve_affine_mul_add_acc(m, modexpr, c_affine, coef.factor);
  }
  return modexpr;
}
denox::memory::optional<SymGraph::value_type>
SymGraph::modsolve_resume_solver(ModSolverHandle solver, symbol lhs, Sym rhs,
                                 uint32_t depth) {
  if (solver == nullptr) {
    return memory::nullopt;
  }
  depth += 1;
  if (depth > MAX_MODSOLVER_DEPTH) {
    DENOX_WARN("symbolic engines modsolver reached max depth. This is "
               "unexpected behaviour, feel free to create a github issue "
               "https://github.com/kistenklaus/denox");
    return memory::nullopt;
  }

  if (rhs.isConstant()) {
    const value_type m = rhs.constant();
    while (solver->expressions.size() <= lhs) {
      symbol s = solver->expressions.size();
      assert(s < m_expressions.size());

      auto expr = m_expressions[s];
      // 1. Take the complete affine expression and check if it collapses
      // under mod rhs.
      denox::memory::optional<value_type> mod = affine_mod(expr.affine, m);
      if (mod.has_value()) {
        ModExpr modexpr;
        modexpr.affine.constant = *mod;
        solver->expressions.push_back(modexpr);
        continue;
      }

      // modsolve_reverse_peel(const AffineExpr &expr, std::size_t m)

      // 2. Derive a affine expression mod rhs by switching over expr.expr
      denox::memory::optional<ModExpr> modexpr;

      switch (expr.expr) {
      case ExprType::Identity:
        break;
      case ExprType::NonAffine: {
        auto nonaffine = m_nonAffineCache.expressions[expr.lhs.sym()];
        switch (nonaffine.expr) {
        case ExprType::Identity:
        case ExprType::NonAffine:
        case ExprType::Add:
        case ExprType::Sub:
          break;
        case symbolic::details::ExprType::Const:
          throw std::runtime_error(
              "Invalid state. ID, NoAffine, Add or Sub are not valid "
              "nonaffine expression types");
          break;
        case ExprType::Div:
          modexpr = modsolve_div(m, nonaffine.symbols[0], nonaffine.symbols[1],
                                 depth);
          break;
        case ExprType::Mod:
          modexpr = modsolve_mod(solver, m, nonaffine.symbols[0],
                                 nonaffine.symbols[1], depth);
          break;
        case ExprType::Min:
        case ExprType::Max:
        case ExprType::Mul:
          // always bail.
          break;
        }
        break;
      }
      case ExprType::Div: {
        modexpr = modsolve_div(m, expr.lhs, expr.rhs, depth);
        break;
      }
      case ExprType::Mod: {
        modexpr = modsolve_mod(solver, m, expr.lhs, expr.rhs, depth);
        break;
      }
      case ExprType::Sub: {
        modexpr = modsolve_sub(solver, m, expr.lhs, expr.rhs);
        break;
      }
      case ExprType::Mul: {
        modexpr = modsolve_mul(solver, m, expr.lhs, expr.rhs);
        break;
      }
      case ExprType::Add: {
        modexpr = modsolve_add(solver, m, expr.lhs, expr.rhs);
        break;
      }
      case ExprType::Const: {
        ModExpr constmodexpr;
        constmodexpr.affine.constant = emod(expr.lhs.constant(), m);
        modexpr = constmodexpr;
        break;
      }
      case ExprType::Min:
      case ExprType::Max:
        // always bail.
        break;
      }

      if (modexpr.has_value()) {
        solver->expressions.push_back(*modexpr);
      } else {
        ModExpr modexprv;
        modexprv.affine.coef.emplace_back(s, 1);
        solver->expressions.push_back(modexprv);
      }
      assert(solver->expressions.size() == s + 1);
    }
  }
  const auto &me = solver->expressions[lhs];
  if (me.affine.isPureConstant())
    return me.affine.constant;
  return denox::memory::nullopt;
}

denox::memory::optional<SymGraph::value_type>
SymGraph::modsolve_reverse_peel(AffineExpr expr, value_type m, uint32_t depth) {

  // NOTE: Find d, such that, there exists a U with, expr = Q div d.
  value_type d = 1;
  for (auto &coef : expr.coef) {
    const auto &c = m_expressions[coef.sym];
    if (c.expr == ExprType::NonAffine) {
      const auto c_nonaffine = m_nonAffineCache.expressions[c.lhs.sym()];
      if (c_nonaffine.expr == ExprType::Div &&
          c_nonaffine.symbols[1].isConstant()) {
        auto num = c_nonaffine.symbols[0];
        auto denom = c_nonaffine.symbols[1].constant();
        auto r = modsolve_resume(num.sym(), Sym::Const(denom), depth);
        if (!r.has_value() || *r != 0)
          return denox::memory::nullopt;
        d = denox::algorithm::lcm(d, denom);
      }
    }
  }
  if (d == 1)
    return denox::memory::nullopt;

  AffineExpr U;
  U.constant = expr.constant * d;
  for (auto coef : expr.coef) {
    const auto c = m_expressions[coef.sym];
    if (c.expr == ExprType::NonAffine) {
      const auto c_nonaffine = m_nonAffineCache.expressions[c.lhs.sym()];
      if (c_nonaffine.expr == ExprType::Div &&
          c_nonaffine.symbols[1].isConstant()) {
        auto num = c_nonaffine.symbols[0];
        auto denom = c_nonaffine.symbols[1].constant();
        value_type scale = coef.factor * (d / denom);
        affine_mul_add_acc(U, m_expressions[num.sym()].affine, scale);
        continue;
      }
    }
    affine_add_sym(U, coef.sym, coef.factor * d);
  }
  {
    assert(m > 0);
    assert(d > 0);

    // LIFTING RULE:
    // If we know r = U mod (m*d), then:
    //   U = k*(m*d) + r   with 0 <= r < m*d
    //   U div d = k*m + floor(r/d)
    // So (U div d) mod m == (floor(r/d) mod m)
    const value_type lifted_mod = m * d;
    auto modexpr =
        modsolve_reduce_affine_mod_m(U, Sym::Const(lifted_mod), depth);
    if (modexpr.has_value()) {
      if (modexpr->affine.isPureConstant()) {
        assert(modexpr->affine.constant < lifted_mod);
        assert(modexpr->affine.constant >= 0);
        const value_type r = modexpr->affine.constant;
        const value_type q = r / d;
        return emod(q, m);
      }
    }
  }
  return d;
}
std::pair<SymGraph::AffineExpr, SymGraph::AffineExpr>
SymGraph::modsolve_mul_only_exact(AffineExpr lhs, Sym rhs, uint32_t depth) {
  AffineExpr Q, R;
  if (!rhs.isConstant()) {
    R = lhs;
    return {Q, R};
  }
  const value_type k = rhs.constant();
  R.constant += lhs.constant;

  for (const auto &coef : lhs.coef) {
    const auto &e = m_expressions[coef.sym];
    if (e.expr == ExprType::NonAffine) {
      const auto &na = m_nonAffineCache.expressions[e.lhs.sym()];
      if (na.expr == ExprType::Div && na.symbols.size() == 2) {
        Sym num = na.symbols[0];
        Sym denom = na.symbols[1];

        if (denom.isConstant() && num.isSymbolic()) {
          value_type c = denom.constant();
          value_type scaled = coef.factor * k;

          if (c == 1) {
            auto d = m_expressions[num.sym()];
            affine_mul_add_acc(Q, d.affine, scaled);
            continue;
          } else if (auto r = modsolve_resume(num.sym(), denom, depth)) {
            if (*r == 0) {
              if ((coef.factor * k) % c == 0) {
                auto d = m_expressions[num.sym()];
                affine_mul_add_acc(Q, d.affine, scaled / c);
                continue;
              }
            }
          }
        }
      }
    }
    R.coef.emplace_back(coef.sym, coef.factor);
  }
  return {Q, R};
}
std::pair<SymGraph::AffineExpr, SymGraph::AffineExpr>
SymGraph::modsolve_peel_by_d(AffineExpr lhs, value_type d, uint32_t depth) {
  assert(d > 0);
  AffineExpr Q, R;

  // Constant: Euclidean split
  {
    auto [qc, rc] = floordivmod(lhs.constant, d);
    Q.constant = qc * d; // keep pre-division multiples in Q
    R.constant = rc;     // 0 <= rc < d
  }

  for (const auto c : lhs.coef) {
    const symbol s = c.sym;
    const value_type f = c.factor;

    // If symbol is known divisible mod d, hoist the entire term to Q.
    if (auto r = modsolve_resume(s, Sym::Const(d), depth); r && *r == 0) {
      Q.coef.push_back({s, f});
      continue;
    }

    // Otherwise do Euclidean split on the coefficient: f = qd + r, 0 <= r < d
    auto [q, rcoef] = floordivmod(f, d);
    if (q != 0) {
      // Put the qd part into Q (still as a multiple-of-d coefficient)
      Q.coef.push_back({s, q * d});
    }
    if (rcoef != 0) {
      // Residual < d goes to R
      R.coef.push_back({s, rcoef});
    }
  }

  return {Q, R};
}

SymGraph::ModExpr SymGraph::modsolve_add(const ModSolverHandle &solver,
                                         value_type m, Sym lhs, Sym rhs) {
  assert(m > 0);
  if (lhs.isConstant() && rhs.isConstant()) {
    ModExpr modexpr;
    modexpr.affine.constant = emod(lhs.constant() + rhs.constant(), m);
    return modexpr;
  }
  AffineExpr a;
  AffineExpr b;
  if (lhs.isConstant() && rhs.isSymbolic()) {
    a.constant = lhs.constant();
    b = solver->expressions[rhs.sym()].affine;
  } else if (rhs.isConstant() && lhs.isSymbolic()) {
    a = solver->expressions[lhs.sym()].affine;
    b.constant = rhs.constant();
  } else {
    a = solver->expressions[lhs.sym()].affine;
    b = solver->expressions[rhs.sym()].affine;
  }
  return modsolve_affine_add(m, a, b);
}
SymGraph::ModExpr SymGraph::modsolve_affine_add(value_type m,
                                                const AffineExpr &lhs,
                                                const AffineExpr &rhs) {
  assert(m > 0);
  ModExpr modexpr;
  modexpr.affine = affine_add(lhs, rhs);
  emod_affine(modexpr.affine, m);
  return modexpr;
}
void SymGraph::modsolve_affine_add_acc(value_type m, ModExpr &lhs,
                                       const AffineExpr &rhs) {
  assert(m > 0);
  affine_add_acc(lhs.affine, rhs);
  emod_affine(lhs.affine, m);
}
void SymGraph::modsolve_affine_mul_add_acc(value_type m, ModExpr &lhs,
                                           const AffineExpr &rhs,
                                           value_type v) {
  assert(m > 0);
  affine_mul_add_acc(lhs.affine, rhs, emod(v, m));
  emod_affine(lhs.affine, m);
}
SymGraph::ModExpr SymGraph::modsolve_sub(const ModSolverHandle &solver,
                                         value_type m, Sym lhs, Sym rhs) {
  assert(m > 0);
  if (lhs.isConstant() && rhs.isConstant()) {
    ModExpr modexpr;
    modexpr.affine.constant = emod(lhs.constant() - rhs.constant(), m);
    return modexpr;
  }
  AffineExpr a;
  AffineExpr b;
  if (lhs.isConstant() && rhs.isSymbolic()) {
    a.constant = lhs.constant();
    b = solver->expressions[rhs.sym()].affine;
  } else if (rhs.isConstant() && lhs.isSymbolic()) {
    a = solver->expressions[lhs.sym()].affine;
    b.constant = rhs.constant();
  } else {
    a = solver->expressions[lhs.sym()].affine;
    b = solver->expressions[rhs.sym()].affine;
  }
  ModExpr modexpr;
  b = affine_mul(b, -1);
  modexpr.affine = affine_add(a, b);
  emod_affine(modexpr.affine, m);
  return modexpr;
}
denox::memory::optional<SymGraph::ModExpr>
SymGraph::modsolve_mul(const ModSolverHandle &solver, value_type m, Sym lhs,
                       Sym rhs) {
  assert(m > 0);
  if (lhs.isConstant() && rhs.isConstant()) {
    ModExpr modexpr;
    modexpr.affine.constant = emod(lhs.constant() * rhs.constant(), m);
    return modexpr;
  }
  AffineExpr a;
  AffineExpr b;
  if (lhs.isConstant() && rhs.isSymbolic()) {
    a.constant = lhs.constant();
    b = solver->expressions[rhs.sym()].affine;
  } else if (rhs.isConstant() && lhs.isSymbolic()) {
    a = solver->expressions[lhs.sym()].affine;
    b.constant = rhs.constant();
  } else {
    return denox::memory::nullopt;
  }
  ModExpr modexpr;
  auto affine = affine_mul(a, b);
  assert(affine.has_value());
  modexpr.affine = *affine;
  emod_affine(modexpr.affine, m);
  return modexpr;
}
denox::memory::optional<SymGraph::ModExpr>
SymGraph::modsolve_div(value_type m, Sym lhs, Sym rhs, uint32_t depth) {
  assert(m > 0);

  // --- Constant / Constant: trivial
  if (lhs.isConstant() && rhs.isConstant()) {
    ModExpr modexpr;
    assert(rhs.constant() > 0);
    modexpr.affine.constant = emod(lhs.constant() / rhs.constant(), m);
    return modexpr;
  }

  // --- Constant / Symbolic: only 0/d collapses; otherwise bail
  if (lhs.isConstant() && rhs.isSymbolic()) {
    if (lhs.constant() == 0) {
      ModExpr out;
      out.affine.constant = 0;
      return out;
    }
    return denox::memory::nullopt;
  }

  // --- Symbolic / Constant: (U div d) mod m
  if (lhs.isSymbolic() && rhs.isConstant()) {
    const value_type d = rhs.constant();
    assert(d > 0);

    // -------- EXACT DIVISION MIRRORING affine_div --------
    // IMPORTANT: use the global ℤ-affine for lhs, not solver->expressions
    // (which is modulo m)
    const AffineExpr U_Z = m_expressions[lhs.sym()].affine;
    if (auto q = affine_div(U_Z, d)) { // your existing exact check
      ModExpr out;
      out.affine = *q;            // quotient in ℤ
      emod_affine(out.affine, m); // reduce coefficients mod m
      return out;                 // pure affine ModExpr, no new symbols
    }

    // LIFTING RULE:
    // If we know r = U mod (m*d), then:
    //   U = k*(m*d) + r   with 0 <= r < m*d
    //   U div d = k*m + floor(r/d)
    // So (U div d) mod m == (floor(r/d) mod m)
    const value_type lifted_mod = m * d;
    modsolve_resume(lhs.sym(), Sym::Const(lifted_mod), depth);
    const auto liftedSolver = require_modsolver(Sym::Const(lifted_mod));
    if (liftedSolver == nullptr ||
        lhs.sym() >= liftedSolver->expressions.size()) {
      // failed to solve lifted expression.
      return memory::nullopt;
    }
    ModExpr lifted = liftedSolver->expressions[lhs.sym()];
    if (lifted.affine.isPureConstant()) {
      value_type r = lifted.affine.constant;
      const value_type q = r / d; // floor since 0 <= r < m*d
      ModExpr out;
      out.affine.constant = emod(q, m); // reduce into Z_m
      return out;
    } else {
      // We have lifted := (U mod (m*d)) as an affine.
      // Compute floor(lifted / d) in ℤ *without* introducing new symbols:
      // peel only the coefficient-qd part; if any variable residual remains,
      // we can't express floor(residual/d) affinely → bail.

      const AffineExpr A = lifted.affine;

      AffineExpr Qdiv;  // quotient part after dividing by d
      AffineExpr Resid; // residual with all coeffs in [0, d)

      // Constant: q_c = floor(A.const / d), r_c in [0,d)
      {
        auto [qc, rc] = floordivmod(A.constant, d);
        Qdiv.constant = qc;
        Resid.constant = rc; // rc < d ⇒ floor(rc/d) == 0
      }

      // Terms: f = q*d + r, with 0 <= r < d
      for (const auto t : A.coef) {
        auto [q, rcoef] = floordivmod(t.factor, d);
        if (q != 0) {
          // contributes q * t.sym to the quotient
          affine_add_sym(Qdiv, t.sym, q);
        }
        if (rcoef != 0) {
          // residual < d; can't take floor(residual/d) without new symbols
          Resid.coef.emplace_back(t.sym, rcoef);
        }
      }

      // If residual has no variable terms, floor(Resid/d) == 0 ⇒ answer is
      // Qdiv
      if (Resid.coef.empty()) {
        ModExpr out;
        out.affine = Qdiv;
        emod_affine(out.affine, m);
        return out;
      }
    }
    // Otherwise, bail.
    return denox::memory::nullopt;
  }
  // --- Symbolic / Symbolic: out of scope for the modsolver now
  return denox::memory::nullopt;
}
denox::memory::optional<SymGraph::ModExpr>
SymGraph::modsolve_mod(const ModSolverHandle &solver, value_type m, Sym lhs,
                       Sym rhs, uint32_t depth) {
  assert(m > 0);
  if (lhs.isConstant() && rhs.isConstant()) {
    ModExpr modexpr;
    assert(rhs.constant() > 0);
    modexpr.affine.constant = emod(emod(lhs.constant(), rhs.constant()), m);
    return modexpr;
  }
  AffineExpr a;
  AffineExpr b;
  if (lhs.isConstant() && rhs.isSymbolic()) {
    a.constant = lhs.constant();
    b = solver->expressions[rhs.sym()].affine;
  } else if (rhs.isConstant() && lhs.isSymbolic()) {
    a = solver->expressions[lhs.sym()].affine;
    if (rhs.constant() % m == 0) {
      return solver->expressions[lhs.sym()];
    }
    b.constant = rhs.constant();
  } else {
    a = solver->expressions[lhs.sym()].affine;
    b = solver->expressions[rhs.sym()].affine;
  }
  // very rarely successful.
  if (auto affine = affine_mod(a, b)) {
    ModExpr modexpr;
    modexpr.affine = *affine;
    emod_affine(modexpr.affine, m);
    return modexpr;
  } else if (lhs.isSymbolic()) {
    if (auto constant = modsolve_resume(lhs.sym(), rhs, depth)) {
      ModExpr modexpr;
      modexpr.affine.constant = emod(*constant, m);
      return modexpr;
    }
  }
  return denox::memory::nullopt;
}

} // namespace denox
