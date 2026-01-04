#include "denox/symbolic/SymGraph.hpp"

namespace denox {

Sym SymGraph::resolve(value_type v) const { return Sym::Const(v); }

Sym SymGraph::resolve(Sym sym) const {
  if (sym.isSymbolic()) {
    auto expr = m_expressions[sym.sym()];
    if (expr.affine.isPureConstant()) {
      return Sym::Const(expr.affine.constant);
    } else {
      return sym;
    }
  } else {
    return Sym::Const(sym.constant());
  }
}

Sym SymGraph::var() { return Sym::Symbol(create_variable(ExprType::Identity)); }

} // namespace denox::compiler
