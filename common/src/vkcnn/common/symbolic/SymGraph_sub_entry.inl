#pragma once
#include "./SymGraph.hpp"

namespace vkcnn {

Sym SymGraph::sub_xx(Sym lhs, Sym rhs, bool dno) {
  if (lhs.isConstant() && rhs.isConstant()) {
    return sub_cc(lhs.constant(), rhs.constant(), dno);
  } else if (lhs.isConstant()) {
    return sub_cs(lhs.constant(), rhs.sym(), dno);
  } else if (rhs.isConstant()) {
    return sub_sc(lhs.sym(), rhs.constant(), dno);
  } else {
    return sub_ss(lhs.sym(), rhs.sym(), dno);
  }
}

Sym SymGraph::sub_ss(symbol lhs, symbol rhs, bool dno) {
  const auto &a = m_expressions[lhs];
  const auto &b = m_expressions[rhs];

  AffineExpr affine = affine_mul(b.affine, value_type(-1));
  affine = affine_add(a.affine, affine);
  return require_affine_sym(ExprType::Sub, Sym::Symbol(lhs), Sym::Symbol(rhs),
                            affine, dno);
}

Sym SymGraph::sub_sc(symbol lhs, value_type rhs, bool dno) {
  const auto &a = m_expressions[lhs];

  AffineExpr affine = a.affine;
  affine.constant -= rhs;
  return require_affine_sym(ExprType::Sub, Sym::Symbol(lhs), Sym::Const(rhs),
                            affine, dno);
}

Sym SymGraph::sub_cs(value_type lhs, symbol rhs, bool dno) {
  const auto &b = m_expressions[rhs];

  AffineExpr affine = b.affine;
  affine = affine_mul(affine, value_type(-1));
  affine.constant += lhs;
  return require_affine_sym(ExprType::Sub, Sym::Const(lhs), Sym::Symbol(rhs),
                            affine, dno);
}

Sym SymGraph::sub_cc(value_type lhs, value_type rhs, bool dno) {
  AffineExpr affine{.coef = {}, .constant = lhs - rhs};
  return require_affine_sym(ExprType::Sub, Sym::Const(lhs), Sym::Const(rhs),
                            affine, dno);
}

} // namespace vkcnn
