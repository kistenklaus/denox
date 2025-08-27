#pragma once
#include "./SymGraph.hpp"

namespace vkcnn {

SymGraph::symbol
SymGraph::require_nonaffine_sym(const NonAffineExpr &nonaffine) {
  assert(nonaffine.expr != ExprType::Sub);
  assert(nonaffine.expr != ExprType::Add);
  symbol nonAffineSym = m_nonAffineCache.expressions.size();
  m_nonAffineCache.expressions.push_back(nonaffine);
  symbolic::details::NonAffineExprKey key{nonAffineSym};
  auto it = m_nonAffineCache.cache.find(key);
  if (it == m_nonAffineCache.cache.end()) {
    m_nonAffineCache.cache.insert(it, std::make_pair(key, nonAffineSym));
    symbol s = create_variable(ExprType::NonAffine);
    m_nonAffineCache.expressions.back().sym = s;
    m_expressions[s].lhs = Sym::Symbol(nonAffineSym);
    return s;
  } else {
    nonAffineSym = it->second;
    m_nonAffineCache.expressions.pop_back();
    return m_nonAffineCache.expressions[nonAffineSym].sym;
  }
}

} // namespace vkcnn
