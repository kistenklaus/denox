#include "denox/symbolic/SymGraph.hpp"

namespace denox {

Sym SymGraph::add_xx(Sym lhs, Sym rhs, bool dno) {
  if (lhs.isConstant() && rhs.isConstant()) {
    return add_cc(lhs.constant(), rhs.constant(), dno);
  } else if (lhs.isConstant()) {
    return add_sc(rhs.sym(), lhs.constant(), dno);
  } else if (rhs.isConstant()) {
    return add_sc(lhs.sym(), rhs.constant(), dno);
  } else {
    return add_ss(lhs.sym(), rhs.sym(), dno);
  }
}
Sym SymGraph::add_ss(symbol lhs, symbol rhs, bool dno) {
  const auto &a = m_expressions[lhs];
  const auto &b = m_expressions[rhs];

  AffineExpr affine = affine_add(a.affine, b.affine);
  return require_affine_sym(ExprType::Add, Sym::Symbol(lhs), Sym::Symbol(rhs),
                            affine, dno);
}
Sym SymGraph::add_sc(symbol lhs, value_type rhs, bool dno) {
  const auto &a = m_expressions[lhs];

  AffineExpr affine = a.affine;
  affine.constant += rhs;
  return require_affine_sym(ExprType::Add, Sym::Symbol(lhs), Sym::Const(rhs),
                            affine, dno);
}
Sym SymGraph::add_cc(value_type lhs, value_type rhs, bool dno) {
  AffineExpr affine{.coef = {}, .constant = lhs + rhs};
  return require_affine_sym(ExprType::Add, Sym::Const(lhs), Sym::Const(rhs),
                            affine, dno);
}

} // namespace vkcnn
