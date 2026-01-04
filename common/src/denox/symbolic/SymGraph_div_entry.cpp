#include "denox/symbolic/SymGraph.hpp"

namespace denox {

Sym SymGraph::div_xx(Sym lhs, Sym rhs, bool dno) {
  if (lhs.isConstant() && rhs.isConstant()) {
    return div_cc(lhs.constant(), rhs.constant(), dno);
  } else if (lhs.isConstant()) {
    return div_cs(lhs.constant(), rhs.sym(), dno);
  } else if (rhs.isConstant()) {
    return div_sc(lhs.sym(), rhs.constant(), dno);
  } else {
    return div_ss(lhs.sym(), rhs.sym(), dno);
  }
}
Sym SymGraph::div_ss(symbol lhs, symbol rhs, bool dno) {
  const auto &a = m_expressions[lhs];
  const auto &b = m_expressions[rhs];
  denox::memory::optional<AffineExpr> affine = affine_div(a.affine, b.affine);
  if (affine.has_value()) {
    return require_affine_sym(ExprType::Div, Sym::Symbol(lhs), Sym::Symbol(rhs),
                              *affine, dno);
  } else {
    return nonaffine_div(Sym::Symbol(lhs), Sym::Symbol(rhs), dno);
  }
}
Sym SymGraph::div_sc(symbol lhs, value_type rhs, bool dno) {
  const auto &a = m_expressions[lhs];
  denox::memory::optional<AffineExpr> affine = affine_div(a.affine, rhs);
  if (affine.has_value()) {
    return require_affine_sym(ExprType::Div, Sym::Symbol(lhs), Sym::Const(rhs),
                              *affine, dno);
  } else {
    return nonaffine_div(Sym::Symbol(lhs), Sym::Const(rhs), dno);
  }
}
Sym SymGraph::div_cs(value_type lhs, symbol rhs, bool dno) {
  if (lhs == 0) {
    AffineExpr affine;
    affine.constant = 0;
    return require_affine_sym(ExprType::Div, Sym::Const(lhs), Sym::Symbol(rhs),
                              affine, dno);
  } else {
    return nonaffine_div(Sym::Const(lhs), Sym::Symbol(rhs), dno);
  }
}
Sym SymGraph::div_cc(value_type lhs, value_type rhs, bool dno) {
  AffineExpr affine;
  assert(rhs > 0);
  affine.constant = lhs / rhs;
  return require_affine_sym(ExprType::Div, Sym::Const(lhs), Sym::Const(rhs),
                            affine, dno);
}

} // namespace vkcnn
