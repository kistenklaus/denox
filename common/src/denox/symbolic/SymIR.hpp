#pragma once

#include "denox/memory/container/span.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/symbolic/SymGraphEval.hpp"
#include <cstdint>

namespace denox {

enum class SymIROpCode {
  Add_SS,
  Add_SC,

  Sub_SS,
  Sub_SC,
  Sub_CS,

  Mul_SS,
  Mul_SC,

  Div_SS,
  Div_SC,
  Div_CS,

  Mod_SS,
  Mod_SC,
  Mod_CS,

  Min_SS,
  Min_SC,

  Max_SS,
  Max_SC,
};

struct SymIROp {
  SymIROpCode opcode;
  std::int64_t lhs;
  std::int64_t rhs;
};

struct SymIR;

class SymIREval {
public:
  friend struct SymIR;
  int64_t operator[](Sym::symbol symbol) const {
    assert(symbol < m_dp.size());
    return m_dp[symbol];
  }

  int64_t operator[](Sym sym) const {
    if (sym.isSymbolic()) {
      assert(sym.sym() < m_dp.size());
      return m_dp[sym.sym()];
    } else {
      return sym.constant();
    }
  }

private:
  SymIREval(memory::vector<int64_t> dp) : m_dp(std::move(dp)) {}
  memory::vector<int64_t> m_dp;
};

struct SymIR {
  SymIREval eval(memory::span<const SymSpec> specs) const;
  std::size_t varCount;
  memory::vector<SymIROp> ops;
};

} // namespace denox
