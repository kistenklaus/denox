#include "symbolic/SymGraph.hpp"
#include "algorithm/gcd.hpp"
#include "algorithm/unstable_sort.hpp"

namespace denox::compiler {

Sym SymGraph::nonaffine_mul(symbol lhs, symbol rhs, bool dno) {
  const Expr a = m_expressions[lhs];
  const Expr b = m_expressions[rhs];

  // Gather pure symbol factors from Identity or NonAffine(Mul) nodes.
  // Hoist any constant factors into `c` (never stored inside Mul.symbols).
  auto gather_mul_factors = [&](symbol s, const Expr &e,
                                denox::memory::small_vector<Sym, 2> &factors,
                                value_type &c) {
    if (e.expr == ExprType::Identity) {
      factors.push_back(Sym::Symbol(s));
      return;
    }
    if (e.expr == ExprType::NonAffine) {
      const auto &na = m_nonAffineCache.expressions[e.lhs.sym()];
      if (na.expr == ExprType::Mul) {
        for (const Sym &t : na.symbols) {
          if (t.isConstant()) {
            c = c * t.constant(); // overflow = UB per your model
          } else {
            factors.push_back(t);
          }
        }
        return;
      }
    }
    // Any other kind: treat as atomic factor
    factors.push_back(Sym::Symbol(s));
  };

  auto build_pure_mul = [&](denox::memory::small_vector<Sym, 2> &syms,
                            value_type c) -> Sym {
    // keep symbols sorted (invariant)
    denox::algorithm::unstable_sort(
        syms.begin(), syms.end(),
        [](const Sym &x, const Sym &y) { return x.sym() < y.sym(); });

    if (syms.empty()) {
      return require_const_sym(c, dno);
    }
    if (syms.size() == 1) {
      if (c == 1)
        return syms[0];
      AffineExpr ae; // c * (single symbol)
      ae.coef.emplace_back(syms[0].sym(), c);
      return require_affine_sym(ExprType::Mul, syms[0], Sym::Const(c), ae, dno);
    }
    NonAffineExpr na; // pure symbolic product, no constants
    na.expr = ExprType::Mul;
    na.symbols = syms;
    Sym prod = Sym::Symbol(require_nonaffine_sym(na));
    if (c == 1)
      return prod;

    // Apply scalar c as an affine coefficient of the product symbol
    AffineExpr ae;
    ae.coef.emplace_back(prod.sym(), c);
    return require_affine_sym(ExprType::Mul, prod, Sym::Const(c), ae, dno);
  };

  const bool a_atom_or_mul =
      (a.expr == ExprType::Identity) || (a.expr == ExprType::NonAffine);
  const bool b_atom_or_mul =
      (b.expr == ExprType::Identity) || (b.expr == ExprType::NonAffine);

  // ---- Case 1: atom/mul × atom/mul → normalize to pure Mul (constants
  // hoisted)
  if (a_atom_or_mul && b_atom_or_mul) {
    denox::memory::small_vector<Sym, 2> syms;
    syms.reserve(4);
    value_type c = 1;
    gather_mul_factors(lhs, a, syms, c);
    gather_mul_factors(rhs, b, syms, c);
    return build_pure_mul(syms, c);
  }

  // Small helper: multiply an affine expression by a single symbol (Identity).
  auto mul_affine_by_symbol = [&](AffineExpr aff, symbol s) -> Sym {
    AffineExpr out; // we'll accumulate as affine where possible
    // constant * symbol → coefficient on that symbol
    if (aff.constant != 0) {
      affine_add_sym(out, s, aff.constant);
    }
    // each linear term: (coef_i * Si) * S  → normalized product P=Si*S, coef_i
    for (const auto &coef : aff.coef) {
      Sym P = nonaffine_mul(coef.sym, s, /*dno*/ false); // normalized product
      if (P.isConstant()) {
        // extremely rare (only happens if coef.sym==S and your mul collapses),
        // but fold if so.
        out.constant += coef.factor * P.constant();
      } else {
        affine_add_sym(out, P.sym(), coef.factor);
      }
    }
    return construct_affine_sym(out, dno);
  };

  // ---- Case 2: affine × symbol (Identity)
  if (b.expr == ExprType::Identity) {
    return mul_affine_by_symbol(a.affine, rhs);
  }

  // ---- Case 3: symbol (Identity) × affine
  if (a.expr == ExprType::Identity) {
    return mul_affine_by_symbol(b.affine, lhs);
  }

  // ---- Case 4: general affine × affine (full distributive)
  AffineExpr out;

  // constant × constant
  out.constant = a.affine.constant * b.affine.constant;

  // linear × constant (both sides)
  if (a.affine.constant != 0) {
    for (const auto coef : b.affine.coef) {
      affine_add_sym(out, coef.sym, coef.factor * a.affine.constant);
    }
  }
  if (b.affine.constant != 0) {
    for (const auto coef : a.affine.coef) {
      affine_add_sym(out, coef.sym, coef.factor * b.affine.constant);
    }
  }

  // linear × linear → normalized products
  for (const auto ac : a.affine.coef) {
    for (const auto bc : b.affine.coef) {
      Sym P = nonaffine_mul(ac.sym, bc.sym, /*dno*/ false);
      value_type k = ac.factor * bc.factor;
      if (P.isConstant()) {
        if (k != 0)
          out.constant += k * P.constant();
      } else {
        affine_add_sym(out, P.sym(), k);
      }
    }
  }

  return require_affine_sym(ExprType::Mul, Sym::Symbol(lhs), Sym::Symbol(rhs),
                            out, dno);
}

Sym SymGraph::nonaffine_div(Sym lhs, Sym rhs, bool dno) {
  // at least one side must be symbolic (both-const handled upstream)
  assert(lhs.isSymbolic() || rhs.isSymbolic());

  // --------------------- Constant / Symbolic ---------------------
  if (lhs.isConstant()) {
    if (lhs.constant() == 0) {
      // 0 / X == 0   (div-by-zero is UB ⇒ X>0)
      return require_const_sym(0, dno);
    }
    // keep as non-affine (e.g. 1/X cannot be folded soundly)
    NonAffineExpr na;
    na.expr = ExprType::Div;
    na.symbols = {lhs, rhs};
    return Sym::Symbol(require_nonaffine_sym(na));
  }

  // --------------------- Symbolic / Constant ---------------------
  if (rhs.isConstant()) {
    assert(lhs.isSymbolic());
    const value_type d = rhs.constant();
    assert(d > 0);
    if (d == 1)
      return lhs; // X / 1 == X

    const auto a = m_expressions[lhs.sym()];

    // helper: (• div c0) div c1 → • div (c0*c1)   |   (c0 div •) div c1 → 0 if
    // c0<c1
    auto propagate_div_by_const = [&](symbol s, value_type d1) -> Sym {
      const auto ae = m_expressions[s];
      const auto na = m_nonAffineCache.expressions[ae.lhs.sym()];
      if (na.expr != ExprType::Div) {
        NonAffineExpr out;
        out.expr = ExprType::Div;
        out.symbols = {Sym::Symbol(s), Sym::Const(d1)};
        return Sym::Symbol(require_nonaffine_sym(out));
      }

      // (U div c0) div c1 => U div (c0*c1)
      if (na.symbols[1].isConstant()) {
        Sym U = na.symbols[0];
        value_type c0 = na.symbols[1].constant();
        value_type c1 = d1;
        value_type div = c0 * c1; // overflow UB matches your model
        NonAffineExpr out;
        out.expr = ExprType::Div;
        out.symbols = {U, Sym::Const(div)};
        return Sym::Symbol(require_nonaffine_sym(out));
      }

      // (c0 div V) div c1 => 0 if c0<c1
      if (na.symbols[0].isConstant()) {
        value_type c0 = na.symbols[0].constant();
        value_type c1 = d1;
        if (c0 < c1)
          return require_const_sym(0, dno);
        NonAffineExpr out;
        out.expr = ExprType::Div;
        out.symbols = {Sym::Const(c0), Sym::Const(c1)};
        return Sym::Symbol(require_nonaffine_sym(out));
      }

      // (U div V) div c => U div (c*V)
      {
        Sym U = na.symbols[0];
        Sym V = na.symbols[1];

        AffineExpr scaled; // represent "c * V" as affine coeff on V
        scaled.coef.emplace_back(V.sym(), d1);
        Sym cV = require_affine_sym(ExprType::Mul, V, Sym::Const(d1), scaled,
                                    /*dno*/ false);

        NonAffineExpr out;
        out.expr = ExprType::Div;
        out.symbols = {U, cV};
        return Sym::Symbol(require_nonaffine_sym(out));
      }
    };

    if (a.expr == ExprType::NonAffine) {
      return propagate_div_by_const(lhs.sym(), d);
    }

    if (a.expr == ExprType::Identity) {
      NonAffineExpr out;
      out.expr = ExprType::Div;
      out.symbols = {lhs, rhs};
      return Sym::Symbol(require_nonaffine_sym(out));
    }

    // -------- affine numerator A / d --------
    {
      // === NEW fast-paths BEFORE any modsolver peeling ===
      if (a.expr != ExprType::NonAffine && a.expr != ExprType::Identity) {
        const auto af = a.affine;

        // Fast path 0: exact split for single-term, no-constant affine
        // (c * Z) div d  →  (c/d) * Z  when c % d == 0
        if (af.constant == 0 && af.coef.size() == 1) {
          const auto c = af.coef[0];
          if (c.factor % d == 0) {
            AffineExpr out;
            affine_add_sym(out, c.sym, c.factor / d);
            return construct_affine_sym(out, dno, lhs.sym());
          }
        }

        // Fast path 1: recognize ceil-div pattern and peel common gcd with d
        // If A == (g*Σti) + (d - 1)  and d % g == 0:
        //   (A) div d  ==  ((Σti) + (d/g - 1)) div (d/g)
        if (!af.coef.empty() && af.constant == d - 1) {
          value_type g = 0;
          for (const auto c : af.coef) {
            value_type f = (c.factor >= 0) ? c.factor : -c.factor;
            g = (g == 0) ? f : denox::algorithm::gcd(g, f);
            if (g == 1)
              break;
          }

          if (g > 1 && (d % g) == 0) {
            // Build Σti by dividing each coefficient by g
            AffineExpr sumTi;
            for (const auto &c : af.coef) {
              affine_add_sym(sumTi, c.sym, c.factor / g);
            }
            Sym sumTiSym =
                construct_affine_sym(sumTi, /*dno*/ false, lhs.sym());

            // Build new numerator: Σti + (d/g - 1)
            value_type d2 = d / g;
            AffineExpr num2 = m_expressions[sumTiSym.sym()].affine; // copy Σti
            num2.constant += (d2 - 1);
            Sym num2Sym =
                construct_affine_sym(num2, /*dno*/ false, sumTiSym.sym());

            // Recurse on strictly smaller divisor
            return nonaffine_div(num2Sym, Sym::Const(d2), dno);
          }
        }
      }
      // === END NEW fast-paths ===

      // 1) If A ≡ 0 (mod d), keep a single Div(A,d)
      if (auto r = modsolve_resume(lhs.sym(), Sym::Const(d)); r && *r == 0) {
        NonAffineExpr out;
        out.expr = ExprType::Div;
        out.symbols = {lhs, Sym::Const(d)};
        return Sym::Symbol(require_nonaffine_sym(out));
      }

      // 2) peel A = Q*d + R  (Q multiples of d; R residual, 0<=R.c<d,
      // residual coeffs in [0,d))
      auto [Q, R] = modsolve_peel_by_d(a.affine, d);

      // 3) Qdiv = Q / d (pure affine)
      AffineExpr Qdiv;
      Qdiv.constant = Q.constant / d;
      Qdiv.coef.reserve(Q.coef.size());
      for (const auto qc : Q.coef) {
        if (qc.factor % d == 0) {
          affine_add_sym(Qdiv, qc.sym, qc.factor / d);
        } else {
          // term was hoisted via S≡0 (mod d); encode sign via +/- Div(|f|*S, d)
          Sym fS = mul_sc(qc.sym, std::abs(qc.factor), /*dno*/ false);
          Sym fSdiv = div_sc(fS.sym(), d, /*dno*/ false);
          affine_add_sym(Qdiv, fSdiv.sym(),
                         (qc.factor < 0) ? value_type(-1) : value_type(1));
        }
      }

      // 4) no residual -> done
      if (R.coef.empty() && R.constant == 0) {
        return construct_affine_sym(Qdiv, dno, lhs.sym());
      }

      // 5) carry-combine micro-optimization for pooling shape:
      //    (Div(U,2) - 1) div 2  ==  Div(U + 2, 4) - 1
      if (d == 2 && R.coef.empty() && R.constant == 1 && Qdiv.constant == 0 &&
          Qdiv.coef.size() == 1) {
        const auto t = Qdiv.coef[0];
        const auto te = m_expressions[t.sym];
        if (t.factor == 1 && te.expr == ExprType::NonAffine) {
          const auto &tna = m_nonAffineCache.expressions[te.lhs.sym()];
          if (tna.expr == ExprType::Div && tna.symbols[1].isConstant() &&
              tna.symbols[1].constant() == 2) {
            Sym U = tna.symbols[0];
            AffineExpr Uplus2 = m_expressions[U.sym()].affine;
            Uplus2.constant += 2;
            Sym num = construct_affine_sym(Uplus2, /*dno*/ false, U.sym());

            NonAffineExpr out;
            out.expr = ExprType::Div;
            out.symbols = {num, Sym::Const(4)};
            Sym divU4 = Sym::Symbol(require_nonaffine_sym(out));
            return sub_xx(divU4, Sym::Const(1), dno);
          }
        }
      }

      // 6) general: Qdiv + Div(R,d); if R is pure constant (<d), Div(R,d)==0
      Sym Qsym = construct_affine_sym(Qdiv, /*dno*/ false, lhs.sym());
      if (R.coef.empty())
        return Qsym; // floor(R/d)==0 when 0<=R<d

      Sym Rsym = construct_affine_sym(R, /*dno*/ false, lhs.sym());
      NonAffineExpr r_na;
      r_na.expr = ExprType::Div;
      r_na.symbols = {Rsym, Sym::Const(d)};
      Sym Rdiv = Sym::Symbol(require_nonaffine_sym(r_na));
      return add_xx(Qsym, Rdiv, dno);
    }
  }

  // --------------------- Symbolic / Symbolic ---------------------
  assert(lhs.isSymbolic() && rhs.isSymbolic());
  assert(lhs.sym() != rhs.sym());

  const auto a = m_expressions[lhs.sym()];
  const auto b = m_expressions[rhs.sym()];

  /* Helper: build a pure symbolic product (no constants).
   * - []      -> Const(1)
   * - [S]     -> S
   * - [S...]  -> NonAffine Mul with symbols sorted by sym()
   */
  auto build_pure_prod = [&](denox::memory::small_vector<Sym, 2> v) -> Sym {
    if (v.empty())
      return Sym::Const(1);
    if (v.size() == 1)
      return v[0];
    std::sort(v.begin(), v.end(),
              [](const Sym &x, const Sym &y) { return x.sym() < y.sym(); });
    symbolic::details::NonAffineExpr na;
    na.expr = ExprType::Mul;
    na.symbols = std::move(v); // symbols only
    return Sym::Symbol(require_nonaffine_sym(na));
  };

  /* Helper: scale any Sym by positive constant k without putting k in a Mul
   * node.
   * - If S is Const(c)      -> Const(c*k)
   * - If S is a symbol node -> affine { coef[(S, k)], constant=0 } with
   * ExprType::Mul
   */
  auto scale_by_const = [&](Sym s, value_type k) -> Sym {
    assert(k > 0);
    if (k == 1)
      return s;
    if (s.isConstant()) {
      return Sym::Const(s.constant() * k); // overflow = UB per your model
    }
    symbolic::details::AffineExpr ae;
    ae.constant = 0;
    ae.coef.emplace_back(s.sym(), k);
    return require_affine_sym(ExprType::Mul, s, Sym::Const(k), ae,
                              /*dno*/ false);
  };

  /* Helper: factorize a symbol-or-Mul node into (const, symbols-only-list) */
  struct Factors {
    value_type c = 1;                         // product of constants (>=1)
    denox::memory::small_vector<Sym, 2> syms; // only symbols, sorted by sym()
  };

  auto factorize_mul = [&](Sym s,
                           const Expr &e) -> denox::memory::optional<Factors> {
    Factors out;

    if (e.expr == ExprType::Identity) {
      out.syms.push_back(s);
      return out;
    }

    if (e.expr == ExprType::NonAffine) {
      const auto na = m_nonAffineCache.expressions[e.lhs.sym()];
      if (na.expr != ExprType::Mul)
        return denox::memory::nullopt;
      for (const Sym &t : na.symbols) {
        if (t.isConstant())
          out.c = out.c * t.constant(); // UB on overflow ok
        else
          out.syms.push_back(t);
      }
      std::sort(out.syms.begin(), out.syms.end(),
                [](const Sym &x, const Sym &y) { return x.sym() < y.sym(); });
      return out;
    }

    // NEW: affine Mul that is exactly k * INNER, with k>0
    if (e.expr == ExprType::Mul) {
      const auto &af = e.affine;
      if (af.constant == 0 && af.coef.size() == 1 && af.coef[0].factor > 0) {
        out.c = af.coef[0].factor; // k > 0
        symbol inner = af.coef[0].sym;
        const Expr &ie = m_expressions[inner];

        if (ie.expr == ExprType::Identity) {
          out.syms.push_back(Sym::Symbol(inner));
        } else if (ie.expr == ExprType::NonAffine) {
          const auto &ina = m_nonAffineCache.expressions[ie.lhs.sym()];
          if (ina.expr == ExprType::Mul) {
            for (const Sym &t : ina.symbols) {
              // by invariant, these are symbols-only already
              out.syms.push_back(t);
            }
          } else {
            // inner is some other non-affine → treat as atomic symbol
            out.syms.push_back(Sym::Symbol(inner));
          }
        } else {
          // inner is affine (rare) → treat as atomic symbol
          out.syms.push_back(Sym::Symbol(inner));
        }

        std::sort(out.syms.begin(), out.syms.end(),
                  [](const Sym &x, const Sym &y) { return x.sym() < y.sym(); });
        return out;
      }
    }

    return denox::memory::nullopt;
  };

  /* 1) (Div(U,V)) / W  ==>  Div(U, V*W)   (V*W stays pure-symbol Mul here) */
  if (a.expr == ExprType::NonAffine) {
    const auto ana = m_nonAffineCache.expressions[a.lhs.sym()];
    if (ana.expr == ExprType::Div) {
      Sym U = ana.symbols[0], V = ana.symbols[1];

      // Make V*rhs as pure symbolic product (no constants inside Mul).
      denox::memory::small_vector<Sym, 2> den_syms = {V, rhs};
      // build_pure_prod sorts symbols and returns either V, rhs, or Mul(V,rhs)
      Sym VW = build_pure_prod(std::move(den_syms));

      symbolic::details::NonAffineExpr out;
      out.expr = ExprType::Div;
      out.symbols = {U, VW}; // ORDERED: numerator, denominator
      return Sym::Symbol(require_nonaffine_sym(out));
    }
  }

  /* 2) product ÷ product: cancel symbolic factors; push constants via affine
   * scaling */
  {
    auto Nf = factorize_mul(lhs, a);
    auto Df = factorize_mul(rhs, b);
    if (Nf && Df) {
      // cancel symbol multisets
      denox::memory::small_vector<Sym, 2> n_rem, d_rem;
      std::size_t i = 0, j = 0;
      while (i < Nf->syms.size() && j < Df->syms.size()) {
        auto ns = Nf->syms[i].sym(), ds = Df->syms[j].sym();
        if (ns < ds)
          n_rem.push_back(Nf->syms[i++]);
        else if (ds < ns)
          d_rem.push_back(Df->syms[j++]);
        else {
          ++i;
          ++j;
        } // cancel 1:1
      }
      for (; i < Nf->syms.size(); ++i)
        n_rem.push_back(Nf->syms[i]);
      for (; j < Df->syms.size(); ++j)
        d_rem.push_back(Df->syms[j]);

      Sym numer_red = build_pure_prod(std::move(n_rem)); // pure symbolic
      Sym denom_red = build_pure_prod(std::move(d_rem)); // pure symbolic

      // fold remaining constant multipliers as affine scaling (not inside Mul)
      if (Nf->c > 1)
        numer_red = scale_by_const(numer_red, Nf->c);
      if (Df->c > 1)
        denom_red = scale_by_const(denom_red, Df->c);

      if (denom_red.isConstant() && denom_red.constant() == 1) {
        return numer_red; // AB/AB → 1; ABC/AB → C; plus constant scalings
      }

      symbolic::details::NonAffineExpr out;
      out.expr = ExprType::Div;
      out.symbols = {numer_red, denom_red}; // ORDERED
      return Sym::Symbol(require_nonaffine_sym(out));
    }
  }

  /* 3) affine numerator ÷ symbolic denominator:
   *    Handles (B*X + B*Y)/(k*B) and (B*X)/(k*B) while keeping pure Mul nodes.
   */
  if (a.expr != ExprType::NonAffine && a.expr != ExprType::Identity) {
    value_type k_const = 1;
    denox::memory::small_vector<Sym, 2> den_syms;

    auto fail_build_div = [&]() -> Sym {
      NonAffineExpr na;
      na.expr = ExprType::Div;
      na.symbols = {lhs, rhs};
      return Sym::Symbol(require_nonaffine_sym(na));
    };

    if (b.expr == ExprType::Identity) {
      den_syms.push_back(rhs);
    } else if (b.expr == ExprType::NonAffine) {
      const auto &bna = m_nonAffineCache.expressions[b.lhs.sym()];
      if (bna.expr != ExprType::Mul)
        return fail_build_div();
      for (const Sym &t : bna.symbols) {
        if (t.isConstant()) {
          value_type v = t.constant();
          assert(v > 0);
          k_const = k_const * v; // UB on overflow ok
        } else {
          den_syms.push_back(t);
        }
      }
      if (den_syms.empty())
        return fail_build_div(); // pure-const handled earlier
    } else if (b.expr == ExprType::Mul) {
      // NEW: accept affine single-term k * INNER, with k>0 and no constant part
      const auto &af = b.affine;
      if (af.constant != 0 || af.coef.size() != 1 || af.coef[0].factor <= 0)
        return fail_build_div();

      k_const = k_const * af.coef[0].factor; // k>0
      symbol inner = af.coef[0].sym;
      const Expr &ie = m_expressions[inner];

      if (ie.expr == ExprType::Identity) {
        den_syms.push_back(Sym::Symbol(inner));
      } else if (ie.expr == ExprType::NonAffine) {
        const auto &ina = m_nonAffineCache.expressions[ie.lhs.sym()];
        if (ina.expr != ExprType::Mul)
          return fail_build_div();
        for (const Sym &t : ina.symbols) {
          // by invariant, symbols-only
          den_syms.push_back(t);
        }
      } else {
        // inner affine → treat as atomic symbol
        den_syms.push_back(Sym::Symbol(inner));
      }
    } else {
      return fail_build_div();
    }

    std::sort(den_syms.begin(), den_syms.end(),
              [](const Sym &x, const Sym &y) { return x.sym() < y.sym(); });

    // Extract symbolic factor multiset from a top-level affine term symbol.
    auto term_syms_of = [&](symbol s)
        -> denox::memory::optional<denox::memory::small_vector<Sym, 2>> {
      const auto &te = m_expressions[s];
      if (te.expr == ExprType::Identity) {
        return denox::memory::small_vector<Sym, 2>{Sym::Symbol(s)};
      }
      if (te.expr == ExprType::NonAffine) {
        const auto &tna = m_nonAffineCache.expressions[te.lhs.sym()];
        if (tna.expr == ExprType::Mul) {
          auto v = tna.symbols; // symbols-only by your invariant
          std::sort(v.begin(), v.end(), [](const Sym &x, const Sym &y) {
            return x.sym() < y.sym();
          });
          return v;
        }
      }
      // Allow a thin affine alias for a single symbol (coef=1, const=0)
      if (te.expr != ExprType::NonAffine && te.expr != ExprType::Identity) {
        const auto &af = te.affine;
        if (af.constant == 0 && af.coef.size() == 1 && af.coef[0].factor == 1) {
          symbol inner = af.coef[0].sym;
          const auto ie = m_expressions[inner];
          if (ie.expr == ExprType::Identity) {
            return denox::memory::small_vector<Sym, 2>{Sym::Symbol(inner)};
          }
          if (ie.expr == ExprType::NonAffine) {
            const auto &ina = m_nonAffineCache.expressions[ie.lhs.sym()];
            if (ina.expr == ExprType::Mul) {
              auto v = ina.symbols;
              std::sort(v.begin(), v.end(), [](const Sym &x, const Sym &y) {
                return x.sym() < y.sym();
              });
              return v;
            }
          }
        }
      }
      return denox::memory::nullopt;
    };

    // A) (Σ B*Ri) / (k*B)  →  (Σ Ri) div k   (requires all terms divisible by B
    // and no const)
    if (den_syms.size() == 1 && k_const > 1 && a.affine.constant == 0) {
      symbol Bsym = den_syms[0].sym();

      bool all_divisible = true;
      symbolic::details::AffineExpr peeled;
      value_type peeled_const = 0;

      for (const auto c : a.affine.coef) {
        auto tvec = term_syms_of(c.sym);
        if (!tvec) {
          all_divisible = false;
          break;
        }

        // remove exactly one B
        bool removed = false;
        denox::memory::small_vector<Sym, 2> rest_syms;
        rest_syms.reserve(tvec->size());
        for (const Sym t : *tvec) {
          if (!removed && t.isSymbolic() && t.sym() == Bsym) {
            removed = true;
            continue;
          }
          rest_syms.push_back(t);
        }
        if (!removed) {
          all_divisible = false;
          break;
        }

        Sym rest =
            build_pure_prod(std::move(rest_syms)); // symbols-only product

        if (rest.isConstant())
          peeled_const += c.factor * rest.constant();
        else
          affine_add_sym(peeled, rest.sym(), c.factor);
      }

      if (all_divisible) {
        peeled.constant += peeled_const;
        Sym sumRest = construct_affine_sym(peeled, /*dno*/ false, lhs.sym());

        symbolic::details::NonAffineExpr out;
        out.expr = ExprType::Div;
        out.symbols = {sumRest, Sym::Const(k_const)}; // (Σ Ri) div k
        return Sym::Symbol(require_nonaffine_sym(out));
      }
      // fall-through
    }

    // B) full peel for Π(den_syms). If k_const>1, require zero constant.
    bool all_divisible = true;
    symbolic::details::AffineExpr Q_full;
    value_type Qfull_const = 0;
    value_type num_const = a.affine.constant;

    for (const auto &c : a.affine.coef) {
      auto tvec = term_syms_of(c.sym);
      if (!tvec) {
        all_divisible = false;
        break;
      }

      if (!includes_multiset(
              std::span<const Sym>(tvec->data(), tvec->size()),
              std::span<const Sym>(den_syms.data(), den_syms.size()))) {
        all_divisible = false;
        break;
      }

      // rest = term / Π den_syms    (symbols-only)
      auto rest_syms =
          multiset_diff(std::span<const Sym>(tvec->data(), tvec->size()),
                        std::span<const Sym>(den_syms.data(), den_syms.size()));

      Sym rest = build_pure_prod(std::move(rest_syms));

      if (rest.isConstant())
        Qfull_const += c.factor * rest.constant();
      else
        affine_add_sym(Q_full, rest.sym(), c.factor);
    }

    if (all_divisible) {
      Q_full.constant += Qfull_const;
      Sym sumRest = construct_affine_sym(Q_full, /*dno*/ false, lhs.sym());

      if (k_const > 1) {
        if (num_const != 0)
          return fail_build_div(); // can’t (soundly) split
        symbolic::details::NonAffineExpr out;
        out.expr = ExprType::Div;
        out.symbols = {sumRest, Sym::Const(k_const)};
        return Sym::Symbol(require_nonaffine_sym(out));
      }

      // k_const == 1
      if (den_syms.size() == 1) {
        if (num_const == 0)
          return sumRest;    // exact
        if (num_const < 0) { // (A*Q - t)/A → Q - 1
          symbolic::details::AffineExpr Qreb = Q_full;
          Qreb.constant -= 1;
          return construct_affine_sym(Qreb, dno, sumRest.sym());
        }
        // num_const > 0: Σ rest + Div(const, A)
        symbolic::details::NonAffineExpr d;
        d.expr = ExprType::Div;
        d.symbols = {Sym::Const(num_const), den_syms[0]}; // ORDERED
        Sym CdivA = Sym::Symbol(require_nonaffine_sym(d));
        return add_xx(sumRest, CdivA, dno);
      }

      // multi-symbol denominator & zero constant → exact Div(sumRest, Π
      // den_syms)
      if (num_const != 0)
        return fail_build_div();
      Sym rhs_prod = build_pure_prod(den_syms);
      symbolic::details::NonAffineExpr out;
      out.expr = ExprType::Div;
      out.symbols = {sumRest, rhs_prod};
      return Sym::Symbol(require_nonaffine_sym(out));
    }

    // C) partial peel for single-symbol divisor (k==1): (Σ A*Ri + R)/A → (Σ Ri)
    // + Div(R,A)
    if (k_const == 1 && den_syms.size() == 1) {
      symbol Asym = den_syms[0].sym();

      symbolic::details::AffineExpr Q_part, R_part;
      R_part.constant = a.affine.constant;
      bool peeled_any = false;

      for (const auto c : a.affine.coef) {
        auto tvec = term_syms_of(c.sym);
        if (!tvec) {
          affine_add_sym(R_part, c.sym, c.factor);
          continue;
        }

        bool removed = false;
        denox::memory::small_vector<Sym, 2> rest_syms;
        rest_syms.reserve(tvec->size());
        for (const Sym t : *tvec) {
          if (!removed && t.isSymbolic() && t.sym() == Asym) {
            removed = true;
            continue;
          }
          rest_syms.push_back(t);
        }

        if (!removed) {
          affine_add_sym(R_part, c.sym, c.factor);
          continue;
        }

        peeled_any = true;
        Sym rest = build_pure_prod(std::move(rest_syms));
        if (rest.isConstant())
          Q_part.constant += c.factor * rest.constant();
        else
          affine_add_sym(Q_part, rest.sym(), c.factor);
      }

      if (peeled_any) {
        // NEW: if residual is a pure negative constant, borrow one denominator:
        // (Q + Div(-t, A)) == (Q - 1) + Div(A - t, A) == (Q - 1) because (A -
        // t) < A
        if (R_part.isPureConstant() && R_part.constant < 0) {
          Q_part.constant -= 1;
          affine_add_sym(R_part, Asym, value_type{1}); // add +1*A to residual
        }

        Sym Qsym = construct_affine_sym(Q_part, /*dno*/ false, lhs.sym());

        // NEW: discharge (A - t)/A == 0  when encoded as 1*A + (negative
        // constant)
        if (R_part.coef.size() == 1 && R_part.coef[0].sym == Asym &&
            R_part.coef[0].factor == 1 && R_part.constant < 0) {
          return Qsym;
        }

        if (R_part.isPureConstant() && R_part.constant == 0)
          return Qsym;

        Sym Rsym = construct_affine_sym(R_part, /*dno*/ false, lhs.sym());
        NonAffineExpr d;
        d.expr = ExprType::Div;
        d.symbols = {Rsym, den_syms[0]}; // ORDERED
        Sym Rdiv = Sym::Symbol(require_nonaffine_sym(d));
        return add_xx(Qsym, Rdiv, dno);
      }
    }

    return fail_build_div();
  }


  /* 4) Fallback: generic Div */
  {
    symbolic::details::NonAffineExpr nonaffine;
    nonaffine.expr = ExprType::Div;
    nonaffine.symbols = {lhs, rhs}; // ORDERED: numerator, denominator
    return Sym::Symbol(require_nonaffine_sym(nonaffine));
  }
}

Sym SymGraph::nonaffine_mod(Sym lhs, Sym rhs, bool dno) {
  // -------------------- both constants --------------------
  if (lhs.isConstant() && rhs.isConstant()) {
    return require_const_sym(emod(lhs.constant(), rhs.constant()), dno);
  }

  // -------------------- const / symbolic --------------------
  if (lhs.isConstant()) {
    assert(rhs.isSymbolic());
    if (lhs.constant() == 0) {
      return require_const_sym(0, dno); // 0 % X == 0
    }
    NonAffineExpr na;
    na.expr = ExprType::Mod;
    na.symbols = {lhs, rhs}; // ORDERED: keep as-is
    return Sym::Symbol(require_nonaffine_sym(na));
  }

  // -------------------- symbolic / const --------------------
  if (rhs.isConstant()) {
    const value_type n = rhs.constant();
    assert(n > 0);
    if (n == 1) {
      return require_const_sym(0, dno); // X % 1 == 0
    }

    const auto le = m_expressions[lhs.sym()];
    if (le.expr == ExprType::NonAffine) {
      const auto lna = m_nonAffineCache.expressions[le.lhs.sym()];
      if (lna.expr == ExprType::Mod && lna.symbols[1].isConstant()) {
        // (X % m) % n  ==>  X % gcd(m, n)
        value_type m = lna.symbols[1].constant();
        value_type g = denox::algorithm::gcd(m, n);
        NonAffineExpr out;
        out.expr = ExprType::Mod;
        out.symbols = {lna.symbols[0], Sym::Const(g)}; // keep order
        return Sym::Symbol(require_nonaffine_sym(out));
      }
    } else if (le.expr == ExprType::Identity) {
      // X % n  stays as a Mod node (cannot fold without more facts)
      NonAffineExpr na;
      na.expr = ExprType::Mod;
      na.symbols = {lhs, rhs};
      return Sym::Symbol(require_nonaffine_sym(na));
    }

    // Reduce affine numerator modulo n (no modsolver facts needed)
    ModExpr amod =
        modsolve_reduce_symbol_mod_m(lhs.sym(), rhs); // keep your path
    AffineExpr affine;
    affine.constant = amod.affine.constant;
    for (const auto coef : amod.affine.coef) {
      value_type f = emod(coef.factor, n);
      if (f != 0) {
        auto c = m_expressions[coef.sym];
        if (c.expr == ExprType::NonAffine) {
          auto c_nonaffine = m_nonAffineCache.expressions[c.lhs.sym()];
          if (c_nonaffine.expr == ExprType::Mod) {
            auto num = c_nonaffine.symbols[0];
            auto denom = c_nonaffine.symbols[1];
            if (denom.isConstant()) {
              // (X % m) % n with m = k*n  ==>  X % n
              value_type m = denom.constant();
              if (m % n == 0) {
                affine_add_sym(affine, num.sym(), f);
                continue;
              }
            }
          }
        }
        affine_add_sym(affine, coef.sym, f);
      }
    }
    affine.constant = emod(affine.constant, n);

    if (affine.isPureConstant()) {
      return require_const_sym(affine.constant, dno);
    }

    Sym num = construct_affine_sym(affine, /*dno*/ false, lhs.sym());
    NonAffineExpr out;
    out.expr = ExprType::Mod;
    out.symbols = {num, Sym::Const(n)}; // ORDERED, do not sort
    return Sym::Symbol(require_nonaffine_sym(out));
  }

  // -------------------- symbolic / symbolic --------------------
  assert(lhs.isSymbolic() && rhs.isSymbolic());

  // Idempotence: (X % Y) % Y == X % Y
  if (lhs.isSymbolic()) {
    const auto le = m_expressions[lhs.sym()];
    if (le.expr == ExprType::NonAffine) {
      const auto &lna = m_nonAffineCache.expressions[le.lhs.sym()];
      if (lna.expr == ExprType::Mod && lna.symbols[1].isSymbolic() &&
          lna.symbols[1].sym() == rhs.sym()) {
        return lhs;
      }
    }
  }

  // A % A == 0
  if (lhs.isSymbolic() && rhs.isSymbolic() && lhs.sym() == rhs.sym()) {
    return require_const_sym(0, dno);
  }

  // Helper: fetch symbol multiset for product nodes (symbols-only per
  // invariant)
  auto as_mul_syms = [&](Sym s, const Expr &e)
      -> denox::memory::optional<denox::memory::small_vector<Sym, 2>> {
    if (e.expr == ExprType::Identity) {
      return denox::memory::small_vector<Sym, 2>{s};
    }
    if (e.expr == ExprType::NonAffine) {
      const auto &na = m_nonAffineCache.expressions[e.lhs.sym()];
      if (na.expr == ExprType::Mul)
        return na.symbols; // already sorted, symbols-only
    }
    return denox::memory::nullopt;
  };

  // Pure product divisibility: (product contains all rhs factors) → 0
  {
    const auto le = m_expressions[lhs.sym()];
    const auto re = m_expressions[rhs.sym()];
    if (auto N = as_mul_syms(lhs, le)) {
      if (auto D = as_mul_syms(rhs, re)) {
        if (includes_multiset(std::span<const Sym>(N->data(), N->size()),
                              std::span<const Sym>(D->data(), D->size()))) {
          return require_const_sym(0, dno);
        }
      }
    }
  }

  // --- symbolic modulus: conservative affine sieve ---
  {
    const auto le = m_expressions[lhs.sym()];
    const auto re = m_expressions[rhs.sym()];

    // Collect denominator factor multiset (symbols-only), already sorted
    denox::memory::small_vector<Sym, 2> den_syms;
    auto collect_den_syms = [&]() -> bool {
      if (re.expr == ExprType::Identity) {
        den_syms.push_back(rhs);
        return true;
      }
      if (re.expr == ExprType::NonAffine) {
        const auto rna = m_nonAffineCache.expressions[re.lhs.sym()];
        if (rna.expr != ExprType::Mul)
          return false;
        den_syms = rna.symbols; // symbols-only (per invariant)
        return true;
      }
      return false; // other rhs shapes → fallback below
    };

    // We only attempt the sieve if numerator is affine (not Identity/NonAffine)
    if (le.expr != ExprType::NonAffine && le.expr != ExprType::Identity) {
      if (collect_den_syms()) {
        // A term is "droppable" iff it is a NonAffine Mul whose factors
        // include ALL modulus factors; Identity terms (like plain A) are NOT
        // droppable.
        auto term_is_droppable = [&](symbol s) -> bool {
          const auto &te = m_expressions[s];
          if (te.expr != ExprType::NonAffine)
            return false;
          const auto &tna = m_nonAffineCache.expressions[te.lhs.sym()];
          if (tna.expr != ExprType::Mul)
            return false;
          return includes_multiset(
              std::span<const Sym>(tna.symbols.data(), tna.symbols.size()),
              std::span<const Sym>(den_syms.data(), den_syms.size()));
        };

        const bool const_is_zero = (le.affine.constant == 0);
        std::size_t droppable = 0, total = le.affine.coef.size();

        for (const auto c : le.affine.coef) {
          if (term_is_droppable(c.sym))
            ++droppable;
        }

        // Only when *every* addend is a droppable product *and* constant==0,
        // we can soundly reduce (sum of multiples) % D -> 0.
        if (const_is_zero && droppable == total) {
          return require_const_sym(0, dno);
        }

        // Otherwise: DO NOT drop partial terms; keep as a generic Mod node.
        // (Prevents factoring sums across %, which your "unsolvable" tests
        // assert.)
      }
    }
  }

  // Fallback: keep as non-affine Mod node (ORDERED; DO NOT SORT)
  {
    NonAffineExpr out;
    out.expr = ExprType::Mod;
    out.symbols = {lhs, rhs};
    return Sym::Symbol(require_nonaffine_sym(out));
  }

  // -------------------- fallback: generic Mod --------------------
  {
    NonAffineExpr out;
    out.expr = ExprType::Mod;
    out.symbols = {lhs, rhs}; // ORDERED; DO NOT SORT
    return Sym::Symbol(require_nonaffine_sym(out));
  }
}

} // namespace denox::compiler
