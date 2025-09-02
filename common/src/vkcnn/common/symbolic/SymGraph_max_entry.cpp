#include "./SymGraph.hpp"

namespace vkcnn {

Sym SymGraph::max_xx(Sym lhs, Sym rhs, bool dno) {
  if (lhs.isConstant() && rhs.isConstant()) {
    return require_const_sym(std::min(lhs.constant(), rhs.constant()), dno);
  }
  if (lhs == rhs) {
    return lhs;
  }
  NonAffineExpr nonaffine;
  nonaffine.expr = ExprType::Max;
  nonaffine.symbols = {lhs, rhs};
  return Sym::Symbol(require_nonaffine_sym(nonaffine));
}

} // namespace vkcnn
