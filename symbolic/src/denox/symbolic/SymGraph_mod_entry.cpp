#include "denox/symbolic/SymGraph.hpp"

namespace denox::compiler {

Sym SymGraph::mod_xx(Sym lhs, Sym rhs, bool dno) {
  if (lhs.isConstant() && rhs.isConstant()) {
    return mod_cc(lhs.constant(), rhs.constant(), dno);
  } else if (lhs.isConstant()) {
    return mod_cs(lhs.constant(), rhs.sym(), dno);
  } else if (rhs.isConstant()) {
    return mod_sc(lhs.sym(), rhs.constant(), dno);
  } else {
    return mod_ss(lhs.sym(), rhs.sym(), dno);
  }
}
Sym SymGraph::mod_ss(symbol lhs, symbol rhs, bool dno) {
  const auto &a = m_expressions[lhs];
  const auto &b = m_expressions[rhs];
  denox::memory::optional<AffineExpr> affine = affine_mod(a.affine, b.affine);
  if (affine.has_value()) {
    return require_affine_sym(ExprType::Mod, Sym::Symbol(lhs), Sym::Symbol(rhs),
                              *affine, dno);
  } else {
    return nonaffine_mod(Sym::Symbol(lhs), Sym::Symbol(rhs), dno);
  }
}
Sym SymGraph::mod_sc(symbol lhs, value_type rhs, bool dno) {
  assert(rhs > 0);
  const auto &a = m_expressions[lhs];
  denox::memory::optional<value_type> mod = affine_mod(a.affine, rhs);
  if (mod.has_value()) {
    return require_const_sym(*mod, dno);
  } else {
    return nonaffine_mod(Sym::Symbol(lhs), Sym::Const(rhs), dno);
  }
}
Sym SymGraph::mod_cs(value_type lhs, symbol rhs, bool dno) {
  if (lhs == 0) {
    AffineExpr affine;
    affine.constant = 0;
    return require_affine_sym(ExprType::Mod, Sym::Const(lhs), Sym::Symbol(rhs),
                              affine, dno);
  } else {
    return nonaffine_mod(Sym::Const(lhs), Sym::Symbol(rhs), dno);
  }
}
Sym SymGraph::mod_cc(value_type lhs, value_type rhs, bool dno) {
  AffineExpr affine;
  assert(rhs > 0);
  affine.constant = lhs % rhs;
  if (affine.constant < 0) {
    affine.constant += rhs;
  }
  return require_affine_sym(ExprType::Mod, Sym::Const(lhs), Sym::Const(rhs),
                            affine, dno);
}

} // namespace vkcnn
