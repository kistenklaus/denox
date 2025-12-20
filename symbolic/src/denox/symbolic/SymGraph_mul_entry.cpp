#include "denox/symbolic/SymGraph.hpp"

namespace denox::compiler {

Sym SymGraph::mul_xx(const Sym lhs, const Sym rhs, const bool dno) {
  if (lhs.isConstant() && rhs.isConstant()) {
    return mul_cc(lhs.constant(), rhs.constant(), dno);
  } else if (lhs.isConstant()) {
    return mul_sc(rhs.sym(), lhs.constant(), dno);
  } else if (rhs.isConstant()) {
    return mul_sc(lhs.sym(), rhs.constant(), dno);
  } else {
    return mul_ss(lhs.sym(), rhs.sym(), dno);
  }
}

Sym SymGraph::mul_ss(symbol lhs, symbol rhs, bool dno) {
  const auto &a = m_expressions[lhs];
  const auto &b = m_expressions[rhs];

  denox::memory::optional<AffineExpr> affine = affine_mul(a.affine, b.affine);
  if (affine.has_value()) {
    return require_affine_sym(ExprType::Mul, Sym::Symbol(lhs), Sym::Symbol(rhs),
                              *affine, dno);
  } else {
    return nonaffine_mul(lhs, rhs, dno);
  }
}

Sym SymGraph::mul_sc(symbol lhs, value_type rhs, bool dno) {
  const auto &a = m_expressions[lhs];

  auto [Q, R] = modsolve_mul_only_exact(a.affine, Sym::Const(rhs));
  R = affine_mul(R, rhs);
  AffineExpr affine = affine_add(Q, R);

  // AffineExpr affine = affine_mul(a.affine, rhs);
  return require_affine_sym(ExprType::Mul, Sym::Symbol(lhs), Sym::Const(rhs),
                            affine, dno);
}

Sym SymGraph::mul_cc(value_type lhs, value_type rhs, bool dno) {
  AffineExpr affine{.coef = {}, .constant = lhs * rhs};
  return require_affine_sym(ExprType::Mul, Sym::Const(lhs), Sym::Const(rhs),
                            affine, dno);
}

} // namespace vkcnn
