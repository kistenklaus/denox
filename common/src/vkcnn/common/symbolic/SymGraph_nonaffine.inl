#pragma once
#include "./SymGraph.hpp"

namespace vkcnn {

Sym SymGraph::nonaffine_mul(symbol lhs, symbol rhs, bool dno) {
  auto a = m_expressions[lhs];
  auto b = m_expressions[rhs];

  static const auto nonaffine_add_sym_associative = [](NonAffineExpr &expr,
                                                       Sym s) {
    auto it = std::lower_bound(expr.symbols.begin(), expr.symbols.end(), s,
                               [](const Sym &a, Sym b) {
                                 if (a.isConstant() && b.isConstant()) {
                                   return a.constant() < b.constant();
                                 } else if (a.isConstant()) {
                                   return true;
                                 } else if (b.isConstant()) {
                                   return false;
                                 } else {
                                   return a.sym() < b.sym();
                                 }
                               });
    expr.symbols.insert(it, s);
  };

  if ((a.expr == ExprType::Identity || a.expr == ExprType::NonAffine) &&
      (b.expr == ExprType::Identity || b.expr == ExprType::NonAffine)) {
    if (a.expr == ExprType::Identity && b.expr == ExprType::Identity) {
      NonAffineExpr nonaffine;
      nonaffine.expr = ExprType::Mul;
      nonaffine_add_sym_associative(nonaffine, Sym::Symbol(lhs));
      nonaffine_add_sym_associative(nonaffine, Sym::Symbol(rhs));
      return Sym::Symbol(require_nonaffine_sym(nonaffine));
    } else if (a.expr != b.expr) {
      if (b.expr == ExprType::Identity) {
        std::swap(a, b);
        std::swap(lhs, rhs);
      }
      assert(a.expr == ExprType::Identity);
      assert(b.expr == ExprType::NonAffine);
      auto b_nonaffine = m_nonAffineCache.expressions[b.lhs.sym()];
      if (b_nonaffine.expr == ExprType::Mul) {
        NonAffineExpr nonaffine;
        nonaffine.expr = ExprType::Mul;
        nonaffine.symbols = b_nonaffine.symbols;
        nonaffine_add_sym_associative(nonaffine, Sym::Symbol(lhs));
        return Sym::Symbol(require_nonaffine_sym(nonaffine));
      }

    } else {
      assert(a.expr == ExprType::NonAffine);
      assert(b.expr == ExprType::NonAffine);
      auto a_nonaffine = m_nonAffineCache.expressions[a.lhs.sym()];
      auto b_nonaffine = m_nonAffineCache.expressions[b.lhs.sym()];
      if (a_nonaffine.expr == ExprType::Mul &&
          b_nonaffine.expr == ExprType::Mul) {
        NonAffineExpr nonaffine;
        nonaffine.expr = ExprType::Mul;
        nonaffine.symbols = a_nonaffine.symbols;
        for (const Sym &sym : b_nonaffine.symbols) {
          nonaffine_add_sym_associative(nonaffine, sym);
        }
        return Sym::Symbol(require_nonaffine_sym(nonaffine));
      }
    }
    NonAffineExpr nonaffine;
    nonaffine.expr = ExprType::Mul;
    nonaffine_add_sym_associative(nonaffine, Sym::Symbol(lhs));
    nonaffine_add_sym_associative(nonaffine, Sym::Symbol(rhs));
    return Sym::Symbol(require_nonaffine_sym(nonaffine));
  }
  // NOTE: Resolve affine expressions into nonaffine or identities.
  // (A + B) * C = AB + BC
  AffineExpr affine;
  // Constant term.
  affine.constant = a.affine.constant * b.affine.constant;

  // Linear tems scaled by the other side's constant.
  if (a.affine.constant != 0) {
    for (const auto &coef : b.affine.coef) {
      affine_add_sym(affine, coef.sym, coef.factor * a.affine.constant);
    }
  }
  if (b.affine.constant != 0) {
    for (const auto &coef : a.affine.coef) {
      affine_add_sym(affine, coef.sym, coef.factor * b.affine.constant);
    }
  }
  for (const auto ac : a.affine.coef) {
    for (const auto bc : b.affine.coef) {
      const Sym sym = nonaffine_mul(ac.sym, bc.sym, dno);
      const value_type k = ac.factor * bc.factor;
      if (sym.isConstant()) {
        const value_type c = sym.constant();
        if (k != 0 && c != 0) {
          // Constant fold!
          affine.constant += k * c;
        }
      } else {
        affine_add_sym(affine, sym.sym(), ac.factor * bc.factor);
      }
    }
  }
  return require_affine_sym(ExprType::Mul, Sym::Symbol(lhs), Sym::Symbol(rhs),
                            affine, dno);
}

Sym SymGraph::nonaffine_div(Sym lhs, Sym rhs, bool dno) {

  static const auto rebuild_product_sum = [&](std::span<const Sym> syms,
                                              bool dno) {
    if (syms.empty()) {
      return require_const_sym(1, dno);
    }
    if (syms.size() == 1) {
      return syms[0];
    }
    NonAffineExpr nonaffine;
    nonaffine.expr = ExprType::Mul;
    nonaffine.symbols.assign(syms.begin(), syms.end());
    return Sym::Symbol(require_nonaffine_sym(nonaffine));
  };

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

    // helper: (• div c0) div c1 → • div (c0*c1)   |   (c0 div •) div c1 → 0
    // if c0<c1
    auto propagate_div_by_const = [&](symbol s, value_type d1) -> Sym {
      const auto &ae = m_expressions[s];
      const auto &na = m_nonAffineCache.expressions[ae.lhs.sym()];
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
      for (const auto &qc : Q.coef) {
        if (qc.factor % d == 0) {
          affine_add_sym(Qdiv, qc.sym, qc.factor / d);
        } else {
          // term was hoisted via S≡0 (mod d); encode sign via +/- Div(|f|*S,
          // d)
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
        const auto &te = m_expressions[t.sym];
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
  assert(lhs.sym() != rhs.sym()); // identical handled upstream in affine_div

  const auto &a = m_expressions[lhs.sym()];
  const auto &b = m_expressions[rhs.sym()];

  // (Div(U,V)) / W  ==>  Div(U, V*W)
  if (a.expr == ExprType::NonAffine) {
    const auto &ana = m_nonAffineCache.expressions[a.lhs.sym()];
    if (ana.expr == ExprType::Div) {
      Sym U = ana.symbols[0], V = ana.symbols[1];

      // build Mul(V, rhs) with sorted symbols
      NonAffineExpr mna;
      mna.expr = ExprType::Mul;
      mna.symbols = {V, rhs};
      std::sort(mna.symbols.begin(), mna.symbols.end(),
                [](const Sym &x, const Sym &y) { return x.sym() < y.sym(); });
      Sym VW = Sym::Symbol(require_nonaffine_sym(mna));

      NonAffineExpr out;
      out.expr = ExprType::Div;
      out.symbols = {U, VW};
      return Sym::Symbol(require_nonaffine_sym(out));
    }
  }

  // ---------- product ÷ product multiset cancellation ----------
  auto as_mul_syms =
      [&](Sym s,
          const Expr &e) -> std::optional<containers::small_vector<Sym, 2>> {
    if (e.expr == ExprType::Identity)
      return containers::small_vector<Sym, 2>{s};
    if (e.expr == ExprType::NonAffine) {
      const auto &na = m_nonAffineCache.expressions[e.lhs.sym()];
      if (na.expr == ExprType::Mul)
        return na.symbols; // symbols only
    }
    return std::nullopt;
  };

  auto build_prod_sym = [&](const containers::small_vector<Sym, 2> &v) -> Sym {
    if (v.empty())
      return Sym::Const(1);
    if (v.size() == 1) {
      AffineExpr a;
      a.coef.emplace_back(v[0].sym(), 1);
      return require_affine_sym(ExprType::Mul, v[0], Sym{}, a, /*dno*/ false);
    }
    NonAffineExpr na;
    na.expr = ExprType::Mul;
    na.symbols = v;
    return Sym::Symbol(require_nonaffine_sym(na));
  };

  if (auto Nopt = as_mul_syms(lhs, a)) {
    if (auto Dopt = as_mul_syms(rhs, b)) {
      const auto &N = *Nopt;
      const auto &D = *Dopt;

      containers::small_vector<Sym, 2> common, n_rem, d_rem;
      {
        std::size_t i = 0, j = 0;
        while (i < N.size() && j < D.size()) {
          auto ns = N[i].sym(), ds = D[j].sym();
          if (ns < ds)
            n_rem.push_back(N[i++]);
          else if (ds < ns)
            d_rem.push_back(D[j++]);
          else {
            common.push_back(N[i++]);
            ++j;
          }
        }
        for (; i < N.size(); ++i)
          n_rem.push_back(N[i]);
        for (; j < D.size(); ++j)
          d_rem.push_back(D[j]);
      }

      if (d_rem.empty()) {
        if (n_rem.empty())
          return Sym::Const(1); // AB/AB
        if (n_rem.size() == 1)
          return n_rem[0]; // ABC/AB -> C
        NonAffineExpr out;
        out.expr = ExprType::Mul;
        out.symbols = std::move(n_rem);
        return Sym::Symbol(require_nonaffine_sym(out));
      }

      if (n_rem.empty()) {
        Sym denom_red = build_prod_sym(d_rem);
        NonAffineExpr out;
        out.expr = ExprType::Div;
        out.symbols = {Sym::Const(1), denom_red};
        return Sym::Symbol(require_nonaffine_sym(out));
      }

      Sym numer_red = build_prod_sym(n_rem);
      Sym denom_red = build_prod_sym(d_rem);
      NonAffineExpr out;
      out.expr = ExprType::Div;
      out.symbols = {numer_red, denom_red};
      return Sym::Symbol(require_nonaffine_sym(out));
    }
  }

  // ---------- affine numerator ÷ symbolic denominator ----------
  if (a.expr != ExprType::NonAffine && a.expr != ExprType::Identity) {
    auto insert_coef_sorted = [](auto &vec, symbol s, value_type f) {
      if (f == 0)
        return;
      auto it = std::lower_bound(
          vec.begin(), vec.end(), s,
          [](const auto &c, symbol key) { return c.sym < key; });
      if (it != vec.end() && it->sym == s) {
        auto nf = it->factor + f;
        if (nf == 0)
          vec.erase(it);
        else
          it->factor = nf;
      } else {
        vec.insert(it, typename std::decay_t<decltype(vec)>::value_type{s, f});
      }
    };

    // collect denominator symbol multiset
    containers::small_vector<Sym, 2> denom_syms;
    if (b.expr == ExprType::Identity) {
      denom_syms.push_back(rhs);
    } else if (b.expr == ExprType::NonAffine) {
      const auto &bna = m_nonAffineCache.expressions[b.lhs.sym()];
      if (bna.expr != ExprType::Mul) {
        NonAffineExpr na;
        na.expr = ExprType::Div;
        na.symbols = {lhs, rhs};
        return Sym::Symbol(require_nonaffine_sym(na));
      }
      denom_syms = bna.symbols;
    } else {
      NonAffineExpr na;
      na.expr = ExprType::Div;
      na.symbols = {lhs, rhs};
      return Sym::Symbol(require_nonaffine_sym(na));
    }

    // split numerator into Q and R per divisibility by denom factor multiset
    AffineExpr Q, R;
    Q.constant = 0;
    R.constant = a.affine.constant;
    value_type pure_den_cancels = 0;

    for (auto const &c : a.affine.coef) {
      const auto &term = m_expressions[c.sym];

      containers::small_vector<Sym, 2> term_syms;
      if (term.expr == ExprType::Identity) {
        term_syms.push_back(Sym::Symbol(c.sym));
      } else if (term.expr == ExprType::NonAffine) {
        const auto &tna = m_nonAffineCache.expressions[term.lhs.sym()];
        if (tna.expr != ExprType::Mul) {
          insert_coef_sorted(R.coef, c.sym, c.factor);
          continue;
        }
        term_syms = tna.symbols;
      } else {
        insert_coef_sorted(R.coef, c.sym, c.factor);
        continue;
      }

      const bool divisible = includes_multiset(
          std::span<const Sym>(term_syms.data(), term_syms.size()),
          std::span<const Sym>(denom_syms.data(), denom_syms.size()));

      if (!divisible) {
        insert_coef_sorted(R.coef, c.sym, c.factor);
        continue;
      }

      auto q_syms = multiset_diff(
          std::span<const Sym>(term_syms.data(), term_syms.size()),
          std::span<const Sym>(denom_syms.data(), denom_syms.size()));

      Sym qsym = rebuild_product_sum(
          std::span<const Sym>(q_syms.data(), q_syms.size()), false);

      if (qsym.isConstant()) {
        Q.constant += c.factor * qsym.constant(); // adds c.factor when qsym==1
        pure_den_cancels += c.factor * qsym.constant();
      } else {
        insert_coef_sorted(Q.coef, qsym.sym(), c.factor);
      }
    }

    // rebalance to enable discharge when residual negative constant but some
    // pure cancellations exist
    if (R.coef.empty() && R.constant < 0 && pure_den_cancels > 0) {
      Q.constant -= 1;
      insert_coef_sorted(R.coef, rhs.sym(), value_type{1});
    }

    // nothing residual → return Q
    if (R.coef.empty() && R.constant == 0) {
      return require_affine_sym(ExprType::Div, lhs, rhs, Q, dno);
    }

    const bool residual_is_sub_rhs_t =
        R.coef.size() == 1 && R.coef[0].sym == rhs.sym() &&
        R.coef[0].factor == 1 &&
        R.constant < 0; // encodes rhs - t with t = -R.constant > 0
    // discharge: (den - t)/den == 0
    if (residual_is_sub_rhs_t) {
      return require_affine_sym(ExprType::Div, lhs, rhs, Q, dno);
    }

    // otherwise: Q + 1 * Div(R, rhs)
    Sym Rsym = require_affine_sym(ExprType::Div, lhs, rhs, R, /*dno*/ false);
    NonAffineExpr dna;
    dna.expr = ExprType::Div;
    dna.symbols = {Rsym, rhs};
    Sym divR = Sym::Symbol(require_nonaffine_sym(dna));

    AffineExpr OUT = Q;
    insert_coef_sorted(OUT.coef, divR.sym(), value_type{1});
    return require_affine_sym(ExprType::Div, lhs, rhs, OUT, dno);
  }

  // Fallback: general non-affine division
  NonAffineExpr nonaffine;
  nonaffine.expr = ExprType::Div;
  nonaffine.symbols = {lhs, rhs};
  return Sym::Symbol(require_nonaffine_sym(nonaffine));
}

Sym SymGraph::nonaffine_mod(Sym lhs, Sym rhs, bool dno) {
  if (lhs.isConstant() && rhs.isConstant()) {
    return require_const_sym(emod(lhs.constant(), rhs.constant()), dno);
  }

  if (lhs.isConstant()) {
    assert(rhs.isSymbolic());
    if (lhs.constant() == 0) {
      return require_const_sym(0, dno);
    }
    NonAffineExpr nonaffine;
    nonaffine.expr = ExprType::Mod;
    nonaffine.symbols = {lhs, rhs};
    return Sym::Symbol(require_nonaffine_sym(nonaffine));
  }

  // ---- constant modulus ----
  if (rhs.isConstant()) {
    const value_type n = rhs.constant();
    assert(n > 0);
    if (n == 1) {
      return require_const_sym(0, dno);
    }

    const auto &le = m_expressions[lhs.sym()];
    if (le.expr == ExprType::NonAffine) {
      const auto &lna = m_nonAffineCache.expressions[le.lhs.sym()];
      if (lna.expr == ExprType::Mod && lna.symbols[1].isConstant()) {
        value_type m = lna.symbols[1].constant();
        value_type g = std::gcd(m, n);
        NonAffineExpr out;
        out.expr = ExprType::Mod;
        out.symbols = {lna.symbols[0], Sym::Const(g)}; // keep order
        return Sym::Symbol(require_nonaffine_sym(out));
      }
    } else if (le.expr == ExprType::Identity) {
      // Early exist.
      NonAffineExpr nonaffine;
      nonaffine.expr = ExprType::Mod;
      nonaffine.symbols = {lhs, rhs};
      return Sym::Symbol(require_nonaffine_sym(nonaffine));
    }

    ModExpr amod = modsolve_reduce_symbol_mod_m(lhs.sym(), rhs);
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
              assert(num.isSymbolic());
              // (X % m) % n
              value_type m = denom.constant();
              if (m % n == 0) {
                // (X % kn) % n = X % n.
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

    Sym num = construct_affine_sym(affine, false, lhs.sym());
    NonAffineExpr out;
    out.expr = ExprType::Mod;
    out.symbols = {num, Sym::Const(n)}; // ORDERED, do not sort
    return Sym::Symbol(require_nonaffine_sym(out));
  }

  assert(lhs.isSymbolic() && rhs.isSymbolic());

  // ---- symbolic modulus ----
  // Idempotence only when same modulus symbol: (X % Y) % Y == X % Y
  if (lhs.isSymbolic()) {
    const auto &le = m_expressions[lhs.sym()];
    if (le.expr == ExprType::NonAffine) {
      const auto &lna = m_nonAffineCache.expressions[le.lhs.sym()];
      if (lna.expr == ExprType::Mod && lna.symbols[1].isSymbolic() &&
          lna.symbols[1].sym() == rhs.sym()) {
        return lhs; // (X % A) % A -> X % A
      }
    }
  }

  // A % A == 0
  if (lhs.isSymbolic() && rhs.isSymbolic() && lhs.sym() == rhs.sym()) {
    return require_const_sym(0, dno);
  }

  // Product divisibility: (product contains all rhs factors) → 0
  auto as_mul_syms =
      [&](Sym s,
          const Expr &e) -> std::optional<containers::small_vector<Sym, 2>> {
    if (e.expr == ExprType::Identity) {
      return containers::small_vector<Sym, 2>{s};
    }
    if (e.expr == ExprType::NonAffine) {
      const auto &na = m_nonAffineCache.expressions[e.lhs.sym()];
      if (na.expr == ExprType::Mul)
        return na.symbols; // symbols only by your invariant
    }
    return std::nullopt;
  };
  if (lhs.isSymbolic() && rhs.isSymbolic()) {
    const auto &le = m_expressions[lhs.sym()];
    const auto &re = m_expressions[rhs.sym()];
    if (auto N = as_mul_syms(lhs, le)) {
      if (auto D = as_mul_syms(rhs, re)) {
        if (includes_multiset(std::span<const Sym>(N->data(), N->size()),
                              std::span<const Sym>(D->data(), D->size()))) {
          return require_const_sym(0, dno);
        }
      }
    }
  }

  // Fallback: keep as non-affine Mod node (ORDERED; DO NOT SORT)
  NonAffineExpr out;
  out.expr = ExprType::Mod;
  out.symbols = {lhs, rhs};
  return Sym::Symbol(require_nonaffine_sym(out));
}

} // namespace vkcnn
