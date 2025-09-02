#pragma once
#include "./SymGraph.hpp"

namespace vkcnn {

vkcnn::Sym vkcnn::SymGraph::resolve(value_type v) const {
  return Sym::Const(v);
}

vkcnn::Sym vkcnn::SymGraph::resolve(Sym sym) const {
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

Sym SymGraph::var()
{
  return Sym::Symbol(create_variable(ExprType::Identity));
}

} // namespace vkcnn
