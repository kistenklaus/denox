#include "denox/symbolic/SymGraphEval.hpp"
#include "denox/symbolic/Symbolic.hpp"

namespace denox::compiler {

memory::optional<Sym::value_type>
SymGraphEval::operator[](const Sym &sym) const {
  if (sym.isConstant()) {
    return sym.constant();
  } else {
    return m_dp[sym.sym()];
  }
}

memory::optional<Sym::value_type> SymGraphEval::operator[](sym s) const {
  if (s.isConstant()) {
    return s.constant();
  } else {
    return m_dp[s.symbol()];
  }
}

memory::optional<Sym::value_type>
SymGraphEval::operator[](const Symbolic &symbolic) const {
  return (*this)[symbolic.resolve()];
}
} // namespace denox::compiler
