#include "denox/diag/unreachable.hpp"
#include "denox/memory/container/hashmap.hpp"
#include "denox/symbolic/SymGraph.hpp"
#include <fmt/format.h>
#include <stdexcept>

namespace denox::compiler {

void SymGraph::debugDump() const {

  for (const auto &[key, solver] : m_modSolverCache) {
    if (key.isConstant()) {
      fmt::println("ModSolve: {}", key.constant());
    } else {
      fmt::println("ModSolve: [{}]", key.sym());
    }
    for (std::size_t s = 0; s < solver->expressions.size(); ++s) {
      auto expr = solver->expressions[s];
      std::string affineStr;
      for (std::size_t c = 0; c < expr.affine.coef.size(); ++c) {
        affineStr += fmt::format("{} * [{}] + ", expr.affine.coef[c].factor,
                                 expr.affine.coef[c].sym);
      }
      affineStr += fmt::format("{}", expr.affine.constant);
      fmt::println("[{}]: {}", s, affineStr);
    }
  }

  fmt::println("NonAffineCache:");
  for (std::size_t e = 0; e < m_nonAffineCache.expressions.size(); ++e) {
    fmt::print("({}) : ", e);
    const auto &expr = m_nonAffineCache.expressions[e];
    const auto form = [](Sym sym) {
      if (sym.isConstant()) {
        return fmt::format("{}", sym.constant());
      } else {
        return fmt::format("[{}]", sym.sym());
      }
    };
    switch (expr.expr) {
    case ExprType::Identity:
      throw std::runtime_error("Invalid state. Identity is affine");
      break;
    case ExprType::NonAffine:
      throw std::runtime_error("Invalid state");
    case ExprType::Div:
      assert(expr.symbols.size() == 2);
      fmt::print("Div({}, {})", form(expr.symbols[0]), form(expr.symbols[1]));
      break;
    case ExprType::Mod:
      assert(expr.symbols.size() == 2);
      fmt::print("Mod({}, {})", form(expr.symbols[0]), form(expr.symbols[1]));
      break;
    case ExprType::Sub:
      throw std::runtime_error("Invalid state. Sub is affine");
    case ExprType::Mul:
      fmt::print("Mul(");
      for (std::size_t c = 0; c < expr.symbols.size(); ++c) {
        if (c != 0) {
          fmt::print(", ");
        }
        fmt::print("{}", form(expr.symbols[c]));
      }
      fmt::print(")");
      break;
    case ExprType::Add:
      throw std::runtime_error("Invalid state. Add is affine");
    case symbolic::details::ExprType::Const:
      throw std::runtime_error("Invalid state. Const is affine");
      break;
    case symbolic::details::ExprType::Min:
      fmt::print("Min(");
      for (std::size_t c = 0; c < expr.symbols.size(); ++c) {
        if (c != 0) {
          fmt::print(", ");
        }
        fmt::print("{}", form(expr.symbols[c]));
      }
      fmt::print(")");
      break;
    case symbolic::details::ExprType::Max:
      fmt::print("Max(");
      for (std::size_t c = 0; c < expr.symbols.size(); ++c) {
        if (c != 0) {
          fmt::print(", ");
        }
        fmt::print("{}", form(expr.symbols[c]));
      }
      fmt::print(")");
      break;
    }
    fmt::println(" -> [{}]", expr.sym);
  }

  fmt::println("Expressions:");
  for (std::size_t e = 0; e < m_expressions.size(); ++e) {
    fmt::print("[{}]: ", e);
    const auto &expr = m_expressions[e];

    const auto &a = expr.lhs;
    std::string avar;
    if (a.isConstant()) {
      avar = fmt::format("{}", a.constant());
    } else {
      avar = fmt::format("[{}]", a.sym());
    }

    const auto &b = expr.rhs;
    std::string bvar;
    if (b.isConstant()) {
      bvar = fmt::format("{}", b.constant());
    } else {
      bvar = fmt::format("[{}]", b.sym());
    }

    std::string exprStr;
    switch (expr.expr) {
    case ExprType::Identity:
      exprStr = fmt::format("Identity");
      break;
    case ExprType::NonAffine:
      exprStr = fmt::format("NonAffine -> ({})", a.sym());
      break;
    case ExprType::Div:
      exprStr = fmt::format("{} div {}", avar, bvar);
      break;
    case ExprType::Mod:
      exprStr = fmt::format("{} mod {}", avar, bvar);
      break;
    case ExprType::Sub:
      exprStr = fmt::format("{} - {}", avar, bvar);
      break;
    case ExprType::Mul:
      exprStr = fmt::format("{} * {}", avar, bvar);
      break;
    case ExprType::Add:
      exprStr = fmt::format("{} + {}", avar, bvar);
      break;
    case symbolic::details::ExprType::Const:
      exprStr = "Const";
      break;
    case symbolic::details::ExprType::Min:
      exprStr = fmt::format("min({},{})", avar, bvar);
      break;
    case symbolic::details::ExprType::Max:
      exprStr = fmt::format("max({} + {})", avar, bvar);
      break;
    }
    std::string affineStr;
    for (std::size_t c = 0; c < expr.affine.coef.size(); ++c) {
      affineStr += fmt::format("{} * [{}] + ", expr.affine.coef[c].factor,
                               expr.affine.coef[c].sym);
    }
    affineStr += fmt::format("{}", expr.affine.constant);

    if (expr.expr == ExprType::NonAffine || expr.expr == ExprType::Identity) {
      fmt::println("{:8}", exprStr);
    } else {
      fmt::println("{:8}  ->   {:10}", exprStr, affineStr);
    }
  }
}

memory::string SymGraph::to_string(
    Sym sym,
    const memory::hash_map<Sym::symbol, memory::string> &symbolNames) const {
  if (sym.isConstant()) {
    return fmt::format("{}", sym.constant());
  }
  const auto &expr = m_expressions[sym.sym()];
  if (expr.affine.isPureConstant()) {
    return fmt::format("{}", expr.affine.constant);
  }
  switch (expr.expr) {
  case symbolic::details::ExprType::Identity:
    if (symbolNames.contains(sym.sym())) {
      return symbolNames.at(sym.sym());
    }
    return fmt::format("[{}]", sym.sym());
  case symbolic::details::ExprType::NonAffine: {
    const auto &nonaffine = m_nonAffineCache.expressions[expr.lhs.sym()];
    switch (nonaffine.expr) {
    case symbolic::details::ExprType::Div:
      return fmt::format("({} / {})", to_string(nonaffine.symbols[0], symbolNames),
                         to_string(nonaffine.symbols[1], symbolNames));
    case symbolic::details::ExprType::Mod:
      return fmt::format("({} % {})", to_string(nonaffine.symbols[0], symbolNames),
                         to_string(nonaffine.symbols[1], symbolNames));
    case symbolic::details::ExprType::Mul: {
      memory::string str = "(";
      for (std::size_t i = 0; i < nonaffine.symbols.size(); ++i) {
        if (i != 0) {
          str.append(" * ");
        }
        str.append(fmt::format("{}", to_string(nonaffine.symbols[i], symbolNames)));
      }
      str.append(")");
      return str;
    }
    case symbolic::details::ExprType::Min:
      return fmt::format("min({}, {})", to_string(expr.lhs, symbolNames), to_string(expr.rhs, symbolNames));
    case symbolic::details::ExprType::Max:
      return fmt::format("max({}, {})", to_string(expr.lhs, symbolNames), to_string(expr.rhs, symbolNames));
    case symbolic::details::ExprType::Identity:
    case symbolic::details::ExprType::NonAffine:
    case symbolic::details::ExprType::Sub:
    case symbolic::details::ExprType::Add:
    case symbolic::details::ExprType::Const:
      diag::unreachable();
    }

    return fmt::format("[NonAffine]");
  }
  case symbolic::details::ExprType::Div:
    return fmt::format("({} / {})", to_string(expr.lhs, symbolNames), to_string(expr.rhs, symbolNames));
  case symbolic::details::ExprType::Mod:
    return fmt::format("({} % {})", to_string(expr.lhs, symbolNames), to_string(expr.rhs, symbolNames));
  case symbolic::details::ExprType::Sub:
    return fmt::format("({} - {})", to_string(expr.lhs, symbolNames), to_string(expr.rhs, symbolNames));
  case symbolic::details::ExprType::Mul:
    return fmt::format("({} * {})", to_string(expr.lhs, symbolNames), to_string(expr.rhs, symbolNames));
  case symbolic::details::ExprType::Add:
    return fmt::format("({} + {})", to_string(expr.lhs, symbolNames), to_string(expr.rhs, symbolNames));
  case symbolic::details::ExprType::Min:
    return fmt::format("min({}, {})", to_string(expr.lhs, symbolNames), to_string(expr.rhs, symbolNames));
  case symbolic::details::ExprType::Max:
    return fmt::format("max({}, {})", to_string(expr.lhs, symbolNames), to_string(expr.rhs, symbolNames));
  case symbolic::details::ExprType::Const:
    return fmt::format("{}", expr.lhs.constant());
  }
  diag::unreachable();
}

} // namespace denox::compiler
