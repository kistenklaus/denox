#pragma once

#include "denox/memory/container/optional.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/symbolic/Sym.hpp"
#include "denox/symbolic/ssym.hpp"

namespace denox {

class Symbolic;

struct SymSpec {
  Sym::symbol symbol;
  Sym::value_type value;
};

class SymGraphEval {
public:
  SymGraphEval() {}
  SymGraphEval(memory::vector<memory::optional<Sym::value_type>> dp)
      : m_dp(std::move(dp)) {}
  memory::optional<Sym::value_type> operator[](const Sym &sym) const;
  memory::optional<Sym::value_type> operator[](sym s) const;
  memory::optional<Sym::value_type> operator[](const Symbolic &s) const;

private:
  memory::vector<memory::optional<Sym::value_type>> m_dp;
};

} // namespace denox::compiler
