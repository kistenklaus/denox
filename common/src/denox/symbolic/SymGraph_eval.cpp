#include "denox/diag/unreachable.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/symbolic/SymGraph.hpp"
#include <cassert>
#include <limits>

namespace denox {

SymGraphEval SymGraph::eval(memory::span<const SymSpec> symSpecs) const {
  memory::vector<memory::optional<Sym::value_type>> dp(m_expressions.size());

  for (const auto &spec : symSpecs) {
    dp[spec.symbol] = spec.value;
  }

  const auto resolve = [&](Sym sym) -> memory::optional<Sym::value_type> {
    if (sym.isConstant()) {
      return sym.constant();
    } else {
      return dp[sym.sym()];
    }
  };

  for (std::size_t e = 0; e < m_expressions.size(); ++e) {
    if (dp[e].has_value()) {
      continue;
    }
    const auto &expr = m_expressions[e];
    switch (expr.expr) {
    case symbolic::details::ExprType::Identity:
      dp[e] = memory::nullopt;
      break;
    case symbolic::details::ExprType::NonAffine: {
      const auto &nonaffine = m_nonAffineCache.expressions[expr.lhs.sym()];
      switch (nonaffine.expr) {
      case symbolic::details::ExprType::Identity:
      case symbolic::details::ExprType::NonAffine:
      case symbolic::details::ExprType::Add:
      case symbolic::details::ExprType::Sub:
      case symbolic::details::ExprType::Const:
        diag::unreachable();
        break;
      case symbolic::details::ExprType::Div: {
        const auto &symbols = nonaffine.symbols;
        assert(symbols.size() == 2);
        const auto num = resolve(symbols[0]);
        const auto denom = resolve(symbols[1]);
        if (num.has_value() && denom.has_value()) {
          dp[e] = num.value() / denom.value();
        }
        break;
      }
      case symbolic::details::ExprType::Mod: {
        const auto &symbols = nonaffine.symbols;
        assert(symbols.size() == 2);
        const auto num = resolve(symbols[0]);
        const auto denom = resolve(symbols[1]);
        if (num.has_value() && denom.has_value()) {
          dp[e] = emod(num.value(), denom.value());
        }
        break;
      }
      case symbolic::details::ExprType::Mul: {
        const auto &symbols = nonaffine.symbols;
        assert(symbols.size() > 1);
        value_type prod = 1;
        bool succ = true;
        for (const auto &fac : symbols) {
          if (const auto opt = resolve(fac)) {
            prod *= *opt;
          } else {
            succ = false;
            break;
          }
        }
        if (succ) {
          dp[e] = prod;
        }
        break;
      }
      case symbolic::details::ExprType::Min: {
        const auto &symbols = nonaffine.symbols;
        assert(symbols.size() > 1);
        value_type min = std::numeric_limits<value_type>::max();
        bool succ = true;
        for (const auto &v : symbols) {
          if (const auto opt = resolve(v)) {
            min = std::min(min, *opt);
          } else {
            succ = false;
            break;
          }
        }
        if (succ) {
          dp[e] = min;
        }
        break;
      }
      case symbolic::details::ExprType::Max: {
        const auto &symbols = nonaffine.symbols;
        assert(symbols.size() > 1);
        value_type max = std::numeric_limits<value_type>::min();
        bool succ = true;
        for (const auto &v : symbols) {
          if (const auto opt = resolve(v)) {
            max = std::max(max, *opt);
          } else {
            succ = false;
            break;
          }
        }
        if (succ) {
          dp[e] = max;
        }
        break;
      }
      }
      break;
    }
    case symbolic::details::ExprType::Div: {
      const auto num = resolve(expr.lhs);
      const auto denom = resolve(expr.rhs);
      if (num.has_value() && denom.has_value()) {
        dp[e] = num.value() / denom.value();
      }
      break;
    }
    case symbolic::details::ExprType::Mod: {
      const auto num = resolve(expr.lhs);
      const auto denom = resolve(expr.rhs);
      if (num.has_value() && denom.has_value()) {
        dp[e] = emod(num.value(), denom.value());
      }
      break;
    }
    case symbolic::details::ExprType::Sub: {
      const auto a = resolve(expr.lhs);
      const auto b = resolve(expr.rhs);
      if (a.has_value() && b.has_value()) {
        dp[e] = a.value() - b.value();
      }
      break;
    }
    case symbolic::details::ExprType::Mul: {
      const auto a = resolve(expr.lhs);
      const auto b = resolve(expr.rhs);
      if (a.has_value() && b.has_value()) {
        dp[e] = a.value() * b.value();
      }
      break;
    }
    case symbolic::details::ExprType::Add: {
      const auto a = resolve(expr.lhs);
      const auto b = resolve(expr.rhs);
      if (a.has_value() && b.has_value()) {
        dp[e] = a.value() + b.value();
      }
      break;
    }

    case symbolic::details::ExprType::Min: {
      const auto a = resolve(expr.lhs);
      const auto b = resolve(expr.rhs);
      if (a.has_value() && b.has_value()) {
        dp[e] = std::min(a.value(), b.value());
      }
      break;
    }
    case symbolic::details::ExprType::Max: {
      const auto a = resolve(expr.lhs);
      const auto b = resolve(expr.rhs);
      if (a.has_value() && b.has_value()) {
        dp[e] = std::max(a.value(), b.value());
      }
      break;
    }
    case symbolic::details::ExprType::Const: {
      assert(expr.lhs.isConstant());
      dp[e] = expr.lhs.constant();
      break;
    }
    }
  }
  return SymGraphEval(dp);
}

} // namespace denox::compiler
