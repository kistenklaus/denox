#include "denox/symbolic/SymGraph.hpp"

namespace denox::compiler {

Sym SymGraph::max_xx(Sym lhs, Sym rhs, bool dno) {
  if (lhs.isConstant() && rhs.isConstant()) {
    return require_const_sym(std::max(lhs.constant(), rhs.constant()), dno);
  }
  if (lhs == rhs) {
    return lhs;
  }
  if (lhs.isConstant() && lhs.constant() == 0) {
    return rhs;
  }
  if (rhs.isConstant() && rhs.constant() == 0) {
    return lhs;
  }

  NonAffineExpr nonaffine;
  nonaffine.expr = ExprType::Max;
  nonaffine.symbols = {lhs, rhs};
  return Sym::Symbol(require_nonaffine_sym(nonaffine));
}

} // namespace denox::compiler
