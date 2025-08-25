#pragma once

#include "vkcnn/common/containers/small_vector.hpp"
#include <algorithm>
#include <cassert>
#include <complex.h>
#include <concepts>
#include <cstdint>
#include <fmt/base.h>
#include <fmt/format.h>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
namespace vkcnn {

class SymGraph;

struct Sym {
  friend SymGraph;

  // NOTE: These are really dangerous during development, but the interface
  // itself it great! operator bool() const { return m_isConstant; }
  // std::int64_t operator*() const { return m_constant; }

  bool isSymbolic() const { return !m_isConstant; }

  std::int64_t value() const {
    assert(m_isConstant);
    return m_constant;
  }
  std::uint64_t sym() const {
    assert(!m_isConstant);
    return m_sym;
  }

  friend bool operator==(const Sym &lhs, const Sym &rhs) {
    if (lhs.isConstant() && rhs.isConstant()) {
      return lhs.constant() == rhs.constant();
    } else if (lhs.isConstant()) {
      return false;
    } else if (rhs.isConstant()) {
      return false;
    } else {
      return lhs.sym() == rhs.sym();
    }
  }

private:
  Sym() : m_isConstant(true), m_constant(0) {}
  explicit Sym(std::int64_t v) : m_isConstant(true), m_constant(v) {}
  explicit Sym(std::uint64_t sym) : m_isConstant(false), m_sym(sym) {}
  static Sym Const(std::int64_t v) { return Sym{v}; }
  static Sym Symbol(std::uint64_t sym) { return Sym{sym}; }

  std::int64_t constant() const {
    assert(m_isConstant);
    return m_constant;
  }
  bool isConstant() const { return m_isConstant; }
  bool m_isConstant;
  union {
    std::uint64_t m_sym;
    std::int64_t m_constant;
  };
};

class SymGraph {
public:
  using symbol = std::uint64_t;
  using value_type = std::int64_t;

  SymGraph() = default;
  SymGraph(const SymGraph &) = delete;
  SymGraph &operator=(const SymGraph &) = delete;

public:
  Sym resolve(Sym sym) const { return reduce_sym(sym); }

  // NOTE: Order in which they are added is important, because it's later
  // used as the parameter index.
  Sym createParameter() {
    return Sym::Symbol(create_variable(ExprType::Identity));
  }

  template <typename L, typename R>
    requires(std::same_as<L, Sym> || std::is_integral_v<L>) &&
            (std::same_as<R, Sym> || std::is_integral_v<R>)
  Sym add(L lhs, R rhs, bool dno = true) {
    Sym a;
    if constexpr (std::same_as<L, Sym>) {
      a = reduce_sym(lhs);
    } else {
      a = Sym::Const(lhs);
    }
    Sym b;
    if constexpr (std::same_as<R, Sym>) {
      b = reduce_sym(rhs);
    } else {
      b = Sym::Const(rhs);
    }
    return add_xx(a, b, dno);
  }

  template <typename L, typename R>
    requires(std::same_as<L, Sym> || std::is_integral_v<L>) &&
            (std::same_as<R, Sym> || std::is_integral_v<R>)
  Sym sub(L lhs, R rhs, bool dno = true) {
    Sym a;
    if constexpr (std::same_as<L, Sym>) {
      a = reduce_sym(lhs);
    } else {
      a = Sym::Const(lhs);
    }
    Sym b;
    if constexpr (std::same_as<R, Sym>) {
      b = reduce_sym(rhs);
    } else {
      b = Sym::Const(rhs);
    }
    return sub_xx(a, b, dno);
  }

  template <typename L, typename R>
    requires(std::same_as<L, Sym> || std::is_integral_v<L>) &&
            (std::same_as<R, Sym> || std::is_integral_v<R>)
  Sym mul(L lhs, R rhs, bool dno = true) {
    Sym a;
    if constexpr (std::same_as<L, Sym>) {
      a = reduce_sym(lhs);
    } else {
      a = Sym::Const(lhs);
    }
    Sym b;
    if constexpr (std::same_as<R, Sym>) {
      b = reduce_sym(rhs);
    } else {
      b = Sym::Const(rhs);
    }
    return mul_xx(a, b, dno);
  }

  template <typename L, typename R>
    requires(std::same_as<L, Sym> || std::is_integral_v<L>) &&
            (std::same_as<R, Sym> || std::is_integral_v<R>)
  Sym div(L lhs, R rhs, bool dno = true) {
    Sym a;
    if constexpr (std::same_as<L, Sym>) {
      a = reduce_sym(lhs);
    } else {
      a = Sym::Const(lhs);
    }
    Sym b;
    if constexpr (std::same_as<R, Sym>) {
      b = reduce_sym(rhs);
    } else {
      b = Sym::Const(rhs);
    }
    return div_xx(a, b, dno);
  }

  template <typename L, typename R>
    requires(std::same_as<L, Sym> || std::is_integral_v<L>) &&
            (std::same_as<R, Sym> || std::is_integral_v<R>)
  Sym mod(L lhs, R rhs = false, bool dno = true) {
    Sym a;
    if constexpr (std::same_as<L, Sym>) {
      a = reduce_sym(lhs);
    } else {
      a = Sym::Const(lhs);
    }
    Sym b;
    if constexpr (std::same_as<R, Sym>) {
      b = reduce_sym(rhs);
    } else {
      b = Sym::Const(rhs);
    }
    return mod_xx(a, b, dno);
  }

  void debug() const {
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

private:
  enum class SymbolType {
    MulSym,
    CeilDivSym,
    FloorDivSym,
    ModSym,
    MaxSym,
    MinSym,
    AlignUpSym,
  };

  struct AffineCoef {
    symbol sym;
    value_type factor;
  };

  struct AffineExpr {
    // NOTE: The coef are always sorted by sym (Invariant)
    containers::small_vector<AffineCoef, 2> coef;
    value_type constant = value_type(0);

    bool isPureConstant() const { return coef.empty(); }
  };

  struct AffineExprHash {
    std::size_t operator()(const AffineExpr &expr) const {
      std::hash<std::uint64_t> hasher;
      std::hash<std::int64_t> ihasher;
      std::size_t hash = 0x1987231298731212;
      for (const auto &coef : expr.coef) {
        hash ^= hasher(coef.sym) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= ihasher(coef.factor) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      }
      hash ^= ihasher(expr.constant) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      return hash;
    }
  };

  struct AffineExprComp {
    bool operator()(const AffineExpr &lhs, const AffineExpr &rhs) const {
      if (lhs.coef.size() != rhs.coef.size()) {
        return false;
      }
      if (lhs.constant != rhs.constant) {
        return false;
      }
      for (std::size_t c = 0; c < lhs.coef.size(); ++c) {
        if ((lhs.coef[c].sym != rhs.coef[c].sym) ||
            (lhs.coef[c].factor != rhs.coef[c].factor)) {
          return false;
        }
      }
      return true;
    }
  };

  using AffineExprCache =
      std::unordered_map<AffineExpr, symbol, AffineExprHash, AffineExprComp>;

  enum class ExprType {
    Identity,  // (a = 0, b = 0)=
    NonAffine, // (a -> nonaffine cache)
               //
    Div,       // a / b
    Mod,       // a % b
    Sub,       // a - b
    Mul,       // a * b
    Add,       // a + b
  };

  struct Expr {
    ExprType expr = ExprType::Identity;
    AffineExpr affine;
    Sym lhs{symbol(0)};
    Sym rhs{symbol(0)};
  };

  struct NonAffineExpr {
    ExprType expr;
    // NOTE: symbols might be a multiset for example 4,X,X,Y is valid!
    // For associative ops symbols are sorted (constants asc, symbols asc)
    containers::small_vector<Sym, 2> symbols;
    symbol sym = symbol(-1);
  };

  struct NonAffineExprKey {
    symbol nonaffinesym;
  };

  using NonAffineExprArray = std::vector<NonAffineExpr>;

  struct NonAffineExprHash {

    NonAffineExprHash(const NonAffineExprArray *expressions)
        : m_exprArrayPtr(expressions) {}

    std::size_t operator()(const NonAffineExprKey &key) const {
      const auto &expr = (*m_exprArrayPtr)[key.nonaffinesym];
      std::hash<std::uint64_t> hasher;
      std::hash<bool> bhasher;
      std::size_t hash = 0x1987231298731212;
      for (const auto &sym : expr.symbols) {
        hash ^=
            bhasher(sym.m_isConstant) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= hasher(sym.m_sym) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      }
      hash ^= hasher(static_cast<std::uint64_t>(
                  static_cast<std::underlying_type_t<ExprType>>(expr.expr))) +
              0x9e3779b9 + (hash << 6) + (hash >> 2);
      return hash;
    }
    const NonAffineExprArray *m_exprArrayPtr;
  };
  struct NonAffineExprComp {
    NonAffineExprComp(const NonAffineExprArray *expressions)
        : m_exprArrayPtr(expressions) {}

    bool operator()(const NonAffineExprKey &lhsKey,
                    const NonAffineExprKey &rhsKey) const {
      const auto &lhs = (*m_exprArrayPtr)[lhsKey.nonaffinesym];
      const auto &rhs = (*m_exprArrayPtr)[rhsKey.nonaffinesym];
      if (lhs.expr != rhs.expr) {
        return false;
      }
      if (lhs.symbols.size() != rhs.symbols.size()) {
        return false;
      }
      for (std::size_t c = 0; c < lhs.symbols.size(); ++c) {
        if (lhs.symbols[c] != rhs.symbols[c]) {
          return false;
        }
      }
      return true;
    }

    const NonAffineExprArray *m_exprArrayPtr;
  };

  struct NonAffineExprCache {

    NonAffineExprCache(const NonAffineExprCache &) = delete;

    NonAffineExprCache()
        : expressions(), cache(0, NonAffineExprHash(&expressions),
                               NonAffineExprComp(&expressions)) {}

    NonAffineExprArray expressions;
    std::unordered_map<NonAffineExprKey, symbol, NonAffineExprHash,
                       NonAffineExprComp>
        cache;
  };

  symbol next_sym() {
    Expr expr;
    symbol s = m_expressions.size();
    m_expressions.emplace_back();
    return s;
  }

  symbol create_variable(ExprType type) {
    assert(type == ExprType::Identity || type == ExprType::NonAffine);
    symbol s = next_sym();
    m_expressions[s].expr = type;
    m_expressions[s].affine.constant = 0;
    m_expressions[s].affine.coef.emplace_back(s, 1);
    m_affineCache.insert(std::make_pair(m_expressions[s].affine, s));
    return s;
  }

  Sym reduce_symbol(symbol s) const {
    const auto &expr = m_expressions[s];
    if (expr.affine.isPureConstant()) {
      return Sym::Const(expr.affine.constant);
    } else {
      return Sym::Symbol(s);
    }
  }

  Sym reduce_sym(Sym sym) const {
    if (sym.isSymbolic()) {
      return reduce_symbol(sym.sym());
    } else {
      return Sym::Const(sym.constant());
    }
  }

  Sym add_xx(Sym lhs, Sym rhs, bool dno = true) {
    if (lhs.isConstant() && rhs.isConstant()) {
      return add_cc(lhs.constant(), rhs.constant(), dno);
    } else if (lhs.isConstant()) {
      return add_sc(rhs.sym(), lhs.constant(), dno);
    } else if (rhs.isConstant()) {
      return add_sc(lhs.sym(), rhs.constant(), dno);
    } else {
      return add_ss(lhs.sym(), rhs.sym(), dno);
    }
  }

  Sym add_ss(symbol lhs, symbol rhs, bool dno) {
    const auto &a = m_expressions[lhs];
    const auto &b = m_expressions[rhs];

    AffineExpr affine = affine_add(a.affine, b.affine);
    return require_affine_sym(ExprType::Add, Sym::Symbol(lhs), Sym::Symbol(rhs),
                              affine, dno);
  }

  Sym add_sc(symbol lhs, value_type rhs, bool dno) {
    const auto &a = m_expressions[lhs];

    AffineExpr affine = a.affine;
    affine.constant += rhs;
    return require_affine_sym(ExprType::Add, Sym::Symbol(lhs), Sym::Const(rhs),
                              affine, dno);
  }

  Sym add_cc(value_type lhs, value_type rhs, bool dno) {
    AffineExpr affine{.coef = {}, .constant = lhs + rhs};
    return require_affine_sym(ExprType::Add, Sym::Const(lhs), Sym::Const(rhs),
                              affine, dno);
  }

  Sym sub_xx(Sym lhs, Sym rhs, bool dno = true) {
    if (lhs.isConstant() && rhs.isConstant()) {
      return sub_cc(lhs.constant(), rhs.constant(), dno);
    } else if (lhs.isConstant()) {
      return sub_cs(lhs.constant(), rhs.sym(), dno);
    } else if (rhs.isConstant()) {
      return sub_sc(lhs.sym(), rhs.constant(), dno);
    } else {
      return sub_ss(lhs.sym(), rhs.sym(), dno);
    }
  }

  Sym sub_ss(symbol lhs, symbol rhs, bool dno) {
    const auto &a = m_expressions[lhs];
    const auto &b = m_expressions[rhs];

    AffineExpr affine = affine_mul(b.affine, value_type(-1));
    affine = affine_add(a.affine, affine);
    return require_affine_sym(ExprType::Sub, Sym::Symbol(lhs), Sym::Symbol(rhs),
                              affine, dno);
  }

  Sym sub_sc(symbol lhs, value_type rhs, bool dno) {
    const auto &a = m_expressions[lhs];

    AffineExpr affine = a.affine;
    affine.constant -= rhs;
    return require_affine_sym(ExprType::Sub, Sym::Symbol(lhs), Sym::Const(rhs),
                              affine, dno);
  }

  Sym sub_cs(value_type lhs, symbol rhs, bool dno) {
    const auto &b = m_expressions[rhs];

    AffineExpr affine = b.affine;
    affine = affine_mul(affine, value_type(-1));
    affine.constant += lhs;
    return require_affine_sym(ExprType::Sub, Sym::Const(lhs), Sym::Symbol(rhs),
                              affine, dno);
  }

  Sym sub_cc(value_type lhs, value_type rhs, bool dno) {
    AffineExpr affine{.coef = {}, .constant = lhs - rhs};
    return require_affine_sym(ExprType::Sub, Sym::Const(lhs), Sym::Const(rhs),
                              affine, dno);
  }

  Sym mul_xx(const Sym lhs, const Sym rhs, const bool dno = true) {
    if (lhs.isConstant() && rhs.isConstant()) {
      return mul_cc(lhs.constant(), rhs.constant(), dno);
    } else if (lhs.isConstant()) {
      return mul_sc(rhs.sym(), lhs.constant(), dno);
    } else if (rhs.isConstant()) {
      return mul_sc(lhs.sym(), rhs.constant(), dno);
    } else {
      return mul_ss(lhs.sym(), rhs.sym(), dno);
    }
  }

  Sym mul_ss(symbol lhs, symbol rhs, bool dno) {
    const auto &a = m_expressions[lhs];
    const auto &b = m_expressions[rhs];

    std::optional<AffineExpr> affine = affine_mul(a.affine, b.affine);
    if (affine.has_value()) {
      return require_affine_sym(ExprType::Mul, Sym::Symbol(lhs),
                                Sym::Symbol(rhs), *affine, dno);
    } else {
      return nonaffine_mul(lhs, rhs, dno);
    }
  }

  Sym mul_sc(symbol lhs, value_type rhs, bool dno) {
    const auto &a = m_expressions[lhs];

    AffineExpr affine = affine_mul(a.affine, rhs);
    return require_affine_sym(ExprType::Mul, Sym::Symbol(lhs), Sym::Const(rhs),
                              affine, dno);
  }

  Sym mul_cc(value_type lhs, value_type rhs, bool dno) {
    AffineExpr affine{.coef = {}, .constant = lhs * rhs};
    return require_affine_sym(ExprType::Mul, Sym::Const(lhs), Sym::Const(rhs),
                              affine, dno);
  }

  Sym div_xx(Sym lhs, Sym rhs, bool dno) {
    if (lhs.isConstant() && rhs.isConstant()) {
      return div_cc(lhs.constant(), rhs.constant(), dno);
    } else if (lhs.isConstant()) {
      return div_cs(lhs.constant(), rhs.sym(), dno);
    } else if (rhs.isConstant()) {
      return div_sc(lhs.sym(), rhs.constant(), dno);
    } else {
      return div_ss(lhs.sym(), rhs.sym(), dno);
    }
  }

  Sym div_ss(symbol lhs, symbol rhs, bool dno) {
    const auto &a = m_expressions[lhs];
    const auto &b = m_expressions[rhs];
    std::optional<AffineExpr> affine = affine_div(a.affine, b.affine);
    if (affine.has_value()) {
      return require_affine_sym(ExprType::Div, Sym::Symbol(lhs),
                                Sym::Symbol(rhs), *affine, dno);
    } else {
      return nonaffine_div(Sym::Symbol(lhs), Sym::Symbol(rhs), dno);
    }
  }

  Sym div_sc(symbol lhs, value_type rhs, bool dno) {
    const auto &a = m_expressions[lhs];
    std::optional<AffineExpr> affine = affine_div(a.affine, rhs);
    if (affine.has_value()) {
      return require_affine_sym(ExprType::Div, Sym::Symbol(lhs),
                                Sym::Const(rhs), *affine, dno);
    } else {
      return nonaffine_div(Sym::Symbol(lhs), Sym::Const(rhs), dno);
    }
  }

  Sym div_cs(value_type lhs, symbol rhs, bool dno) {
    if (lhs == 0) {
      AffineExpr affine;
      affine.constant = 0;
      return require_affine_sym(ExprType::Div, Sym::Const(lhs),
                                Sym::Symbol(rhs), affine, dno);
    } else {
      return nonaffine_div(Sym::Const(lhs), Sym::Symbol(rhs), dno);
    }
  }

  Sym div_cc(value_type lhs, value_type rhs, bool dno) {
    AffineExpr affine;
    assert(rhs > 0);
    affine.constant = lhs / rhs;
    return require_affine_sym(ExprType::Div, Sym::Const(lhs), Sym::Const(rhs),
                              affine, dno);
  }

  Sym mod_xx(Sym lhs, Sym rhs, bool dno) {
    if (lhs.isConstant() && rhs.isConstant()) {
      return mod_cc(lhs.constant(), rhs.constant(), dno);
    } else if (lhs.isConstant()) {
      return mod_cs(lhs.constant(), rhs.sym(), dno);
    } else if (rhs.isConstant()) {
      return mod_sc(lhs.sym(), rhs.constant(), dno);
    } else {
      return mod_ss(lhs.sym(), rhs.sym(), dno);
    }
  }

  Sym mod_ss(symbol lhs, symbol rhs, bool dno) {
    const auto &a = m_expressions[lhs];
    const auto &b = m_expressions[rhs];
    std::optional<AffineExpr> affine = affine_mod(a.affine, b.affine);
    if (affine.has_value()) {
      return require_affine_sym(ExprType::Mod, Sym::Symbol(lhs),
                                Sym::Symbol(rhs), *affine, dno);
    } else {
      return nonaffine_mod(Sym::Symbol(lhs), Sym::Symbol(rhs), dno);
    }
  }

  Sym mod_sc(symbol lhs, value_type rhs, bool dno) {
    const auto &a = m_expressions[lhs];
    std::optional<AffineExpr> affine = affine_mod(a.affine, rhs);
    if (affine.has_value()) {
      return require_affine_sym(ExprType::Mod, Sym::Symbol(lhs),
                                Sym::Const(rhs), *affine, dno);
    } else {
      std::optional<value_type> mod = modsolve_resume(lhs, Sym::Const(rhs));
      if (mod.has_value()) {
        return require_const_sym(ExprType::Mod, Sym::Symbol(lhs),
                                 Sym::Const(rhs), mod.value(), dno);
      }
      return nonaffine_mod(Sym::Symbol(lhs), Sym::Const(rhs), dno);
    }
  }

  Sym mod_cs(value_type lhs, symbol rhs, bool dno) {
    if (lhs == 0) {
      AffineExpr affine;
      affine.constant = 0;
      return require_affine_sym(ExprType::Mod, Sym::Const(lhs),
                                Sym::Symbol(rhs), affine, dno);
    } else {
      return nonaffine_mod(Sym::Const(lhs), Sym::Symbol(rhs), dno);
    }
  }

  Sym mod_cc(value_type lhs, value_type rhs, bool dno) {
    AffineExpr affine;
    assert(rhs > 0);
    affine.constant = lhs % rhs;
    if (affine.constant < 0) {
      affine.constant += rhs;
    }
    return require_affine_sym(ExprType::Mod, Sym::Const(lhs), Sym::Const(rhs),
                              affine, dno);
  }

  Sym require_affine_sym(ExprType type, Sym lhs, Sym rhs,
                         const AffineExpr &affine, bool dno) {
    assert(type != ExprType::NonAffine);
    if (!dno && affine.isPureConstant()) {
      return Sym::Const(affine.constant);
    }

    const auto [s, exists] = requireSymForAffine(affine);
    if (exists) {
      return Sym::Symbol(s);
    }
    m_expressions[s].expr = type;
    m_expressions[s].affine = affine;
    m_expressions[s].lhs = lhs;
    m_expressions[s].rhs = rhs;
    return Sym::Symbol(s);
  }

  Sym require_const_sym(ExprType type, Sym lhs, Sym rhs, value_type constant,
                        bool dno) {
    if (!dno) {
      return Sym::Const(constant);
    }
    AffineExpr affine;
    affine.constant = constant;
    return require_affine_sym(type, lhs, rhs, affine, dno);
  }

  std::pair<symbol, bool> requireSymForAffine(const AffineExpr &expr) {
    auto it = m_affineCache.find(expr);
    if (it == m_affineCache.end()) {
      symbol s = next_sym();
      m_affineCache.insert(it, std::make_pair(expr, s));
      return {s, false};
    } else {
      return {it->second, true};
    }
  }

  static void affine_add_sym(AffineExpr &lhs, symbol s, value_type factor) {
    if (factor == 0)
      return;

    auto &v = lhs.coef;

    auto it = std::lower_bound(
        v.begin(), v.end(), s,
        [](const AffineCoef &a, symbol b) { return a.sym < b; });

    if (it != v.end() && it->sym == s) {
      const value_type new_factor = it->factor + factor;
      if (new_factor == 0) {
        v.erase(it);
      } else {
        it->factor = new_factor;
      }
      return;
    }

    v.insert(it, AffineCoef{s, factor});
  }

  AffineExpr affine_add(const AffineExpr &lhs, const AffineExpr &rhs) {
    AffineExpr out;
    out.constant = lhs.constant + rhs.constant;
    out.coef = lhs.coef;
    for (std::size_t bx = 0; bx < rhs.coef.size(); ++bx) {
      const auto &coef = rhs.coef[bx];
      affine_add_sym(out, coef.sym, coef.factor);
    }
    return out;
  }

  AffineExpr affine_mul(const AffineExpr &lhs, const value_type &rhs) {
    if (rhs == 0) {
      AffineExpr affine;
      affine.constant = 0;
      return affine;
    }
    AffineExpr out = lhs;
    for (std::size_t c = 0; c < out.coef.size(); ++c) {
      out.coef[c].factor *= rhs;
    }
    out.constant *= rhs;
    return out;
  }

  std::optional<AffineExpr> affine_mul(const AffineExpr &lhs,
                                       const AffineExpr &rhs) {
    // NOTE: Only affine is one side is constant,
    // otherwise we require terms like X*Y or X^2, which are nonaffine.
    if (lhs.coef.empty()) {
      return affine_mul(rhs, lhs.constant);
    }
    if (rhs.coef.empty()) {
      return affine_mul(lhs, rhs.constant);
    }
    return std::nullopt;
  }
  // Returns true iff A == k * B (all coeffs + constant), and writes k_out.
  // No allocations; quadratic scan is fine because coef lists are tiny.
  static std::optional<value_type>
  find_affine_scalar_multiple(const AffineExpr &a, const AffineExpr &b) {
    bool similar = a.coef.size() == b.coef.size();
    for (std::size_t ax = 0; ax < a.coef.size() && similar; ++ax) {
      bool exists = false;
      for (std::size_t bx = 0; bx < b.coef.size(); ++bx) {
        if (a.coef[ax].sym == b.coef[bx].sym) {
          exists = true;
          break;
        }
      }
      if (!exists) {
        similar = false;
      }
    }
    if (!similar) {
      return std::nullopt;
    }
    assert(a.coef.size() == b.coef.size());
    // Denominator can't be the zero expression.
    if (b.coef.empty() && b.constant == 0)
      return std::nullopt;

    bool have_k = false;
    value_type k = 0;

    // If B has a constant, fix k from constants.
    if (b.constant != 0) {
      if (a.constant % b.constant != 0)
        return std::nullopt;
      k = a.constant / b.constant;
      have_k = true;
    } else {
      // Then A's constant must be 0 as well.
      if (a.constant != 0)
        return std::nullopt;
    }

    // For each term in B, find matching term in A and check the ratio.
    std::size_t matched = 0;
    for (const auto &bt : b.coef) {
      // Find bt.sym in A
      const AffineCoef *atp = nullptr;
      for (const auto &at : a.coef) {
        if (at.sym == bt.sym) {
          atp = &at;
          break;
        }
      }
      if (!atp)
        return std::nullopt; // missing symbol

      // Determine or verify k
      if (!have_k) {
        if (bt.factor == 0)
          return std::nullopt; // shouldn't happen in canonical form
        if (atp->factor % bt.factor != 0)
          return std::nullopt;
        k = atp->factor / bt.factor;
        have_k = true;
      } else {
        if (atp->factor != k * bt.factor)
          return std::nullopt;
      }
      ++matched;
    }

    // Ensure A has no extra symbols beyond B
    if (matched != a.coef.size())
      return std::nullopt;

    return k;
  }

  template <class T> static inline std::pair<T, T> floordivmod(T a, T b) {
    assert(b > 0);
    T q = a / b;
    T r = a % b;
    if (r < 0) {
      --q;
      r += b;
    }
    return {q, r};
  }

  std::optional<AffineExpr> affine_div(const AffineExpr &lhs, value_type rhs) {
    assert(rhs > 0);
    if (rhs == 1)
      return lhs;

    // All variable coefficients must be divisible by rhs
    AffineExpr out;
    out.coef.reserve(lhs.coef.size());
    for (auto const &c : lhs.coef) {
      if (c.factor % rhs != 0) {
        return std::nullopt; // non-affine residual -> bail
      }
      out.coef.push_back({c.sym, c.factor / rhs});
    }

    // Constant term uses floor division
    auto [q, _] = floordivmod<value_type>(lhs.constant, rhs);
    out.constant = q;

    return out;
  }

  std::optional<AffineExpr> affine_div(const AffineExpr &lhs,
                                       const AffineExpr &rhs) {
    if (rhs.coef.empty()) {
      assert(rhs.constant > 0);
      return affine_div(lhs, rhs.constant);
    }

    if (lhs.coef.empty()) {
      if (lhs.constant == 0) {
        AffineExpr affine;
        affine.constant = 0;
        return affine;
      }
      return std::nullopt;
    }

    if (lhs.coef.size() == rhs.coef.size()) {
      bool equalSymbols = true;
      for (std::size_t c = 0; c < lhs.coef.size() && equalSymbols; ++c) {
        if (lhs.coef[c].sym != rhs.coef[c].sym) {
          equalSymbols = false;
        }
      }
      if (equalSymbols) {

        // NOTE: A / B = (k(A')+r) div h(B'), where
        // k = gcd(A-coefficients)
        // A' = A / k
        // r = A-const mod k
        // h = gcd(B)
        // B' = B / h
        //
        // If A' == B', then this collapses into:
        // k(A') div hB' + r div hB'
        value_type k = 0;
        for (const auto &coef : lhs.coef) {
          k = std::gcd(k, std::abs(coef.factor));
        }
        if (lhs.coef.empty()) {
          k = 1;
        }
        AffineExpr A = lhs;
        for (auto &coef : A.coef) {
          coef.factor /= k;
        }
        A.constant = lhs.constant / k;
        value_type r = lhs.constant % k;
        // if (r < 0) {
        //   r += k;
        // }

        value_type h = rhs.constant;
        for (const auto &coef : rhs.coef) {
          h = std::gcd(h, std::abs(coef.factor));
        }
        AffineExpr B = rhs;
        for (auto &coef : B.coef) {
          coef.factor /= h;
        }
        B.constant /= h;

        bool AeqB = A.constant == B.constant;
        for (std::size_t c = 0; c < A.coef.size() && AeqB; ++c) {
          if (A.coef[c].factor != B.coef[c].factor) {
            AeqB = false;
          }
        }
        if (AeqB) {
          value_type kh = k / h;

          if (r == 0) {
            AffineExpr affine;
            affine.constant = kh;
            return affine;
          } else if (r < 0 && -r <= h) {
            AffineExpr affine;
            affine.constant = kh - 1;
            return affine;
          } else if (r > 0 && r < h) {
            AffineExpr affine;
            affine.constant = kh;
            return affine;
          }
        }
      }
    }

    if (lhs.coef.size() == 1 && rhs.coef.size() == 1 &&
        lhs.coef.front().sym == rhs.coef.front().sym &&
        lhs.constant == rhs.constant) {
      // Single symbol cancels single symbol
      AffineExpr out;
      auto a = lhs.coef.front();
      auto b = rhs.coef.front();
      assert(a.sym == b.sym);
      out.constant = a.factor / b.factor;
      return out;
    }

    return std::nullopt;
  }

  std::optional<AffineExpr> affine_mod(const AffineExpr &lhs,
                                       const value_type &rhs) {
    assert(rhs > 0);
    if (rhs == 1) {
      AffineExpr affine;
      affine.constant = 0;
      return affine;
    }
    if (lhs.isPureConstant()) {
      assert(lhs.constant >= 0);
      AffineExpr affine;
      affine.constant = lhs.constant % rhs;
      if (affine.constant < 0) {
        affine.constant += rhs;
      }
      return affine;
    }

    bool allDiv = true;
    for (auto &coef : lhs.coef) {
      if (coef.factor % rhs != 0) {
        allDiv = false;
        break;
      }
    }
    if (allDiv) {
      AffineExpr affine;
      affine.constant = lhs.constant % rhs;
      if (affine.constant < 0) {
        affine.constant += rhs;
      }
      return affine;
    }
    return std::nullopt;
  }

  std::optional<AffineExpr> affine_mod(const AffineExpr &lhs,
                                       const AffineExpr &rhs) {
    if (lhs.isPureConstant()) {
      assert(lhs.constant >= 0);
      if (rhs.isPureConstant()) {
        assert(rhs.constant > 0);
        AffineExpr affine;
        affine.constant = lhs.constant % rhs.constant;
        if (affine.constant < 0) {
          affine.constant += rhs.constant;
        }
        return affine;
      } else {
        if (lhs.constant == 0) {
          AffineExpr affine;
          affine.constant = 0;
          return affine;
        }
        return std::nullopt;
      }
    } else {
      if (rhs.isPureConstant()) {
        assert(rhs.constant > 0);
        return affine_mod(lhs, rhs.constant);
      } else {
        assert(!lhs.isPureConstant() && !rhs.isPureConstant());
        const static auto full_gcd = [](const AffineExpr &e) -> value_type {
          value_type g = 0;
          for (auto &t : e.coef)
            g = std::gcd(g, static_cast<value_type>(std::abs(t.factor)));
          g = std::gcd(g, static_cast<value_type>(std::abs(e.constant)));
          return g == 0 ? 1 : g; // avoid 0
        };
        const static auto divide_all = [](AffineExpr e, value_type k) {
          for (auto &t : e.coef)
            t.factor /= static_cast<value_type>(k);
          e.constant /= static_cast<value_type>(k);
          return e;
        };

        const static auto affine_equal = [](const AffineExpr &a,
                                            const AffineExpr &b) {
          if (a.constant != b.constant || a.coef.size() != b.coef.size())
            return false;
          for (size_t i = 0; i < a.coef.size(); ++i)
            if (a.coef[i].sym != b.coef[i].sym ||
                a.coef[i].factor != b.coef[i].factor)
              return false;
          return true;
        };
        {
          value_type gL = full_gcd(lhs);
          value_type gR = full_gcd(rhs);
          AffineExpr EL = divide_all(lhs, gL);
          AffineExpr ER = divide_all(rhs, gR);
          if (affine_equal(EL, ER)) {
            value_type s = gL % gR;
            if (s < 0) {
              s += gR;
            }
            AffineExpr out;
            for (auto &t : EL.coef) {
              value_type v = t.factor * static_cast<value_type>(s);
              if (v != 0) {
                out.coef.emplace_back(t.sym, v);
              }
            }
            out.constant = EL.constant * static_cast<value_type>(s);
            return out;
          }
        }
      }
    }
    return std::nullopt;
  }

  // NOTE: iff. false the Sym already exists returns 0 and does not modify
  // expr.
  void nonaffine_add_sym_associative(NonAffineExpr &expr, Sym s) {
    auto it = std::lower_bound(expr.symbols.begin(), expr.symbols.end(), s,
                               [](const Sym &a, Sym b) {
                                 if (a.isConstant() && b.isConstant()) {
                                   return a.constant() < b.constant();
                                 } else if (a.isConstant()) {
                                   return true;
                                 } else if (b.isConstant()) {
                                   return false;
                                 } else {
                                   return a.sym() < b.sym();
                                 }
                               });
    expr.symbols.insert(it, s);
  }

  // NOTE reduces a nonaffine expression into a affine by introducing new
  // symbols.
  Sym nonaffine_mul(symbol lhs, symbol rhs, bool dno) {
    auto a = m_expressions[lhs];
    auto b = m_expressions[rhs];

    if ((a.expr == ExprType::Identity || a.expr == ExprType::NonAffine) &&
        (b.expr == ExprType::Identity || b.expr == ExprType::NonAffine)) {
      if (a.expr == ExprType::Identity && b.expr == ExprType::Identity) {
        NonAffineExpr nonaffine;
        nonaffine.expr = ExprType::Mul;
        nonaffine_add_sym_associative(nonaffine, Sym::Symbol(lhs));
        nonaffine_add_sym_associative(nonaffine, Sym::Symbol(rhs));
        return Sym::Symbol(require_nonaffine_sym(nonaffine));
      } else if (a.expr != b.expr) {
        if (b.expr == ExprType::Identity) {
          std::swap(a, b);
          std::swap(lhs, rhs);
        }
        assert(a.expr == ExprType::Identity);
        assert(b.expr == ExprType::NonAffine);
        auto b_nonaffine = m_nonAffineCache.expressions[b.lhs.sym()];
        if (b_nonaffine.expr == ExprType::Mul) {
          NonAffineExpr nonaffine;
          nonaffine.expr = ExprType::Mul;
          nonaffine.symbols = b_nonaffine.symbols;
          nonaffine_add_sym_associative(nonaffine, Sym::Symbol(lhs));
          return Sym::Symbol(require_nonaffine_sym(nonaffine));
        }

      } else {
        assert(a.expr == ExprType::NonAffine);
        assert(b.expr == ExprType::NonAffine);
        auto a_nonaffine = m_nonAffineCache.expressions[a.lhs.sym()];
        auto b_nonaffine = m_nonAffineCache.expressions[b.lhs.sym()];
        if (a_nonaffine.expr == ExprType::Mul &&
            b_nonaffine.expr == ExprType::Mul) {
          NonAffineExpr nonaffine;
          nonaffine.expr = ExprType::Mul;
          nonaffine.symbols = a_nonaffine.symbols;
          for (const Sym &sym : b_nonaffine.symbols) {
            nonaffine_add_sym_associative(nonaffine, sym);
          }
          return Sym::Symbol(require_nonaffine_sym(nonaffine));
        }
      }
      NonAffineExpr nonaffine;
      nonaffine.expr = ExprType::Mul;
      nonaffine_add_sym_associative(nonaffine, Sym::Symbol(lhs));
      nonaffine_add_sym_associative(nonaffine, Sym::Symbol(rhs));
      return Sym::Symbol(require_nonaffine_sym(nonaffine));
    }
    // NOTE: Resolve affine expressions into nonaffine or identities.
    // (A + B) * C = AB + BC
    AffineExpr affine;
    // Constant term.
    affine.constant = a.affine.constant * b.affine.constant;

    // Linear tems scaled by the other side's constant.
    if (a.affine.constant != 0) {
      for (const auto &coef : b.affine.coef) {
        affine_add_sym(affine, coef.sym, coef.factor * a.affine.constant);
      }
    }
    if (b.affine.constant != 0) {
      for (const auto &coef : a.affine.coef) {
        affine_add_sym(affine, coef.sym, coef.factor * b.affine.constant);
      }
    }
    for (const auto ac : a.affine.coef) {
      for (const auto bc : b.affine.coef) {
        const Sym sym = nonaffine_mul(ac.sym, bc.sym, dno);
        const value_type k = ac.factor * bc.factor;
        if (sym.isConstant()) {
          const value_type c = sym.constant();
          if (k != 0 && c != 0) {
            // Constant fold!
            affine.constant += k * c;
          }
        } else {
          affine_add_sym(affine, sym.sym(), ac.factor * bc.factor);
        }
      }
    }
    return require_affine_sym(ExprType::Mul, Sym::Symbol(lhs), Sym::Symbol(rhs),
                              affine, dno);
  }

  // Split an affine A as A = d*Q + R, with 0 <= R.constant < d and
  // for every symbol coefficient c: c = d*(c/d) + (c%d)
  static inline void split_affine_by_const(const AffineExpr &A, value_type d,
                                           AffineExpr &Q, AffineExpr &R) {
    assert(d > 0);
    Q.constant = 0;
    R.constant = 0;
    Q.coef.clear();
    Q.coef.reserve(A.coef.size());
    R.coef.clear();
    R.coef.reserve(A.coef.size());

    for (auto const &c : A.coef) {
      // Coefficients should be non-negative in your IR; document with an
      // assert.
      assert(c.factor >= 0);
      value_type q = c.factor / d;
      value_type r = c.factor % d; // d>0 ⇒ r in [0..d-1] for nonneg factors
      if (q)
        Q.coef.emplace_back(c.sym, q);
      if (r)
        R.coef.emplace_back(c.sym, r);
    }
    auto [qc, rc] = floordivmod(A.constant, d); // Euclidean: 0 <= rc < d
    Q.constant = qc;
    R.constant = rc;
  }

  // True if denom_syms ⊆ term_syms (with multiplicity). Both must be sorted.
  static inline bool includes_multiset(std::span<const Sym> term_syms,
                                       std::span<const Sym> denom_syms) {
    std::size_t i = 0, j = 0;
    while (i < term_syms.size() && j < denom_syms.size()) {
      if (term_syms[i].sym() < denom_syms[j].sym()) {
        ++i;
      } else if (denom_syms[j].sym() < term_syms[i].sym()) {
        return false;
      } else {
        ++i;
        ++j; // equal -> consume one of each
      }
    }
    return j == denom_syms.size();
  }

  // term_syms - denom_syms (precondition: denom_syms ⊆ term_syms). Both sorted.
  static inline containers::small_vector<Sym, 2>
  multiset_diff(std::span<const Sym> term_syms,
                std::span<const Sym> denom_syms) { // can be const too
    containers::small_vector<Sym, 2> out;
    std::size_t i = 0, j = 0;
    while (i < term_syms.size() && j < denom_syms.size()) {
      if (term_syms[i].sym() < denom_syms[j].sym()) {
        out.push_back(term_syms[i++]);
      } else if (denom_syms[j].sym() < term_syms[i].sym()) {
        // Should not happen under the subset precondition; advance denom to be
        // safe.
        ++j;
      } else {
        // equal -> cancel one occurrence
        ++i;
        ++j;
      }
    }
    for (; i < term_syms.size(); ++i)
      out.push_back(term_syms[i]);
    return out;
  }

  // Build a symbol for a product of already-sorted factors (constants/symbols).
  // Returns: 1 if empty; identity if single symbol; NonAffine Mul otherwise.
  inline Sym rebuild_product_sum(std::span<const Sym> syms, Sym lhs, Sym rhs,
                                 bool dno) {
    if (syms.empty()) {
      return require_const_sym(ExprType::Mul, lhs, rhs, 1, dno);
    }
    if (syms.size() == 1) {
      AffineExpr a;
      a.coef.emplace_back(syms[0].sym(), 1);
      return require_affine_sym(ExprType::Mul, syms[0], Sym{}, a,
                                /*dno*/ false);
    }
    NonAffineExpr na;
    na.expr = ExprType::Mul;
    na.symbols.assign(syms.begin(), syms.end());
    return Sym::Symbol(require_nonaffine_sym(na));
  }

  // true iff R has the exact affine shape: 1*rhs + (-t) with t>0 and no other
  // terms
  inline bool residual_is_sub_rhs_t(const AffineExpr &R, symbol rhs) {
    if (R.coef.size() != 1)
      return false;
    if (R.coef[0].sym != rhs || R.coef[0].factor != 1)
      return false;
    return (R.constant < 0); // represents rhs - t with t = -R.constant > 0
  }

  // (Optional) strip constant factors from a product multiset (denominator
  // side)
  inline containers::small_vector<Sym, 2>
  filter_symbol_factors(std::span<const Sym> syms) {
    containers::small_vector<Sym, 2> out;
    for (auto const &s : syms)
      if (!s.isConstant())
        out.push_back(s);
    return out;
  }

  Sym nonaffine_div(Sym lhs, Sym rhs, bool dno) {
    // either lhs or rhs is symbolic (both-const handled by affine_div upstream)
    assert(lhs.isSymbolic() || rhs.isSymbolic());

    // --------------------- Constant / Symbolic ---------------------
    if (lhs.isConstant()) {
      if (lhs.constant() == 0) {
        // 0 / X == 0   (X is symbolic; div-by-zero is UB so X>0)
        return require_const_sym(ExprType::Div, lhs, rhs, 0, dno);
      }
      // leave as non-affine (don't fold 1/X -> 0; that is unsound if X==1)
      NonAffineExpr na;
      na.expr = ExprType::Div;
      na.symbols = {lhs, rhs};
      return Sym::Symbol(require_nonaffine_sym(na));
    }

    // --------------------- Symbolic / Constant ---------------------
    if (rhs.isConstant()) {
      assert(lhs.isSymbolic());
      const value_type d = rhs.constant();
      assert(d > 0);
      if (d == 1)
        return lhs; // X / 1 == X

      const auto &a = m_expressions[lhs.sym()];
      if (a.expr == ExprType::NonAffine) {
        // (X div c0) div c1 → X div (c0*c1); (c0 div X) div c1 → 0 if c0 < c1
        const auto nonaffine_div_constant = [&](symbol lhs_sym,
                                                value_type d1) -> Sym {
          const auto &ae = m_expressions[lhs_sym];
          const auto &na = m_nonAffineCache.expressions[ae.lhs.sym()];
          if (na.expr == ExprType::Div) {
            if (na.symbols[1].isConstant()) {
              // (U div c0) div c1 => U div (c0*c1)
              Sym U = na.symbols[0];
              value_type c0 = na.symbols[1].constant();
              value_type c1 = d1;
              value_type div = c0 * c1; // UB on overflow is fine in your model
              NonAffineExpr out;
              out.expr = ExprType::Div;
              out.symbols = {U, Sym::Const(div)};
              return Sym::Symbol(require_nonaffine_sym(out));
            } else if (na.symbols[0].isConstant()) {
              // (c0 div V) div c1 => 0  if c0 < c1
              value_type c0 = na.symbols[0].constant();
              value_type c1 = d1;
              if (c0 < c1) {
                return require_const_sym(ExprType::Div, Sym::Symbol(lhs_sym),
                                         Sym::Const(d1), 0, dno);
              }
            } else {
              // NEW: (U div V) div c  =>  U div (c * V)
              Sym U = na.symbols[0];
              Sym V = na.symbols[1];

              // Build affine symbol for (c * V). No constants inside Mul nodes.
              AffineExpr scaled;
              scaled.coef.emplace_back(V.sym(),
                                       d1); // coefficient d1 on symbol V
              Sym cV = require_affine_sym(ExprType::Mul, V, Sym::Const(d1),
                                          scaled, /*dno*/ false);

              NonAffineExpr out;
              out.expr = ExprType::Div;
              out.symbols = {U, cV};
              return Sym::Symbol(require_nonaffine_sym(out));
            }
          }
          NonAffineExpr out;
          out.expr = ExprType::Div;
          out.symbols = {Sym::Symbol(lhs_sym), Sym::Const(d1)};
          return Sym::Symbol(require_nonaffine_sym(out));
        };

        return nonaffine_div_constant(lhs.sym(), d);
      }

      if (a.expr == ExprType::Identity) {
        NonAffineExpr out;
        out.expr = ExprType::Div;
        out.symbols = {lhs, rhs};
        return Sym::Symbol(require_nonaffine_sym(out));
      }

      // a is affine: split A = d*Q + R (Euclidean on constant), return Q +
      // div(R,d)
      {
        AffineExpr Q, R;
        split_affine_by_const(a.affine, d, Q, R);

        if (R.coef.empty()) {
          // R is a pure constant in [0,d), floor(R/d)==0
          return require_affine_sym(ExprType::Div, lhs, rhs, Q, dno);
        }

        Sym Rsym =
            require_affine_sym(ExprType::Div, lhs, rhs, R, /*dno*/ false);
        NonAffineExpr dna;
        dna.expr = ExprType::Div;
        dna.symbols = {Rsym, rhs};
        Sym divR = Sym::Symbol(require_nonaffine_sym(dna));

        AffineExpr OUT = Q;
        // local inserter
        auto insert_coef_sorted2 = [](auto &vec, symbol s, value_type f) {
          if (f == 0)
            return;
          auto it = std::lower_bound(
              vec.begin(), vec.end(), s,
              [](const auto &c, symbol key) { return c.sym < key; });
          if (it != vec.end() && it->sym == s) {
            auto nf = it->factor + f;
            if (nf == 0)
              vec.erase(it);
            else
              it->factor = nf;
          } else {
            vec.insert(it,
                       typename std::decay_t<decltype(vec)>::value_type{s, f});
          }
        };
        insert_coef_sorted2(OUT.coef, divR.sym(), value_type{1});
        return require_affine_sym(ExprType::Div, lhs, rhs, OUT, dno);
      }
    }

    // --------------------- Symbolic / Symbolic ---------------------
    assert(lhs.isSymbolic() && rhs.isSymbolic());

    // A/A handled by affine_div upstream (identical symbols), so:
    assert(lhs.sym() != rhs.sym());

    const auto &a = m_expressions[lhs.sym()];
    const auto &b = m_expressions[rhs.sym()];

    // (Div(U,V)) / W  ==>  Div(U, V*W)     (all symbolic, >0)
    if (a.expr == ExprType::NonAffine) {
      const auto &ana = m_nonAffineCache.expressions[a.lhs.sym()];
      if (ana.expr == ExprType::Div) {
        Sym U = ana.symbols[0];
        Sym V = ana.symbols[1];

        // Build product V*rhs (Mul)
        NonAffineExpr mna;
        mna.expr = ExprType::Mul;
        mna.symbols = {V, rhs};
        std::sort(mna.symbols.begin(), mna.symbols.end(),
                  [](const Sym &x, const Sym &y) { return x.sym() < y.sym(); });

        Sym VW = Sym::Symbol(require_nonaffine_sym(mna));
        NonAffineExpr out;
        out.expr = ExprType::Div;
        out.symbols = {U, VW};
        return Sym::Symbol(require_nonaffine_sym(out));
      }
    }

    // -------- Product ÷ Product GCD cancellation (covers AB/AC → B/C, AB/AB →
    // 1, A/AB → 1/B) --------
    auto build_prod_sym =
        [&](const containers::small_vector<Sym, 2> &v) -> Sym {
      if (v.empty())
        return Sym::Const(1);
      if (v.size() == 1) {
        AffineExpr a;
        a.coef.emplace_back(v[0].sym(), 1);
        return require_affine_sym(ExprType::Mul, v[0], Sym{}, a, /*dno*/ false);
      }
      NonAffineExpr na;
      na.expr = ExprType::Mul;
      na.symbols = v;
      return Sym::Symbol(require_nonaffine_sym(na));
    };

    // If both numerator and denominator are “symbolic products” (Identity or
    // Mul), do multiset GCD.
    auto as_mul_syms =
        [&](Sym s,
            const Expr &e) -> std::optional<containers::small_vector<Sym, 2>> {
      if (e.expr == ExprType::Identity) {
        return containers::small_vector<Sym, 2>{s};
      }
      if (e.expr == ExprType::NonAffine) {
        const auto &na = m_nonAffineCache.expressions[e.lhs.sym()];
        if (na.expr == ExprType::Mul) {
          // per your invariant, these contain only symbols (no constants)
          return na.symbols;
        }
      }
      return std::nullopt;
    };

    if (auto Nopt = as_mul_syms(lhs, a)) {
      if (auto Dopt = as_mul_syms(rhs, b)) {
        const auto &N = *Nopt;
        const auto &D = *Dopt;

        // Compute multiset intersection and remainders
        containers::small_vector<Sym, 2> common, n_rem, d_rem;
        {
          std::size_t i = 0, j = 0;
          while (i < N.size() && j < D.size()) {
            auto ns = N[i].sym(), ds = D[j].sym();
            if (ns < ds) {
              n_rem.push_back(N[i++]);
            } else if (ds < ns) {
              d_rem.push_back(D[j++]);
            } else {
              common.push_back(N[i++]);
              ++j;
            }
          }
          for (; i < N.size(); ++i)
            n_rem.push_back(N[i]);
          for (; j < D.size(); ++j)
            d_rem.push_back(D[j]);
        }

        if (d_rem.empty()) {
          // exact division
          if (n_rem.empty())
            return Sym::Const(1); // AB/AB
          if (n_rem.size() == 1)
            return n_rem[0]; // ABC/AB → C (identity)
          NonAffineExpr out;
          out.expr = ExprType::Mul;
          out.symbols = std::move(n_rem);
          return Sym::Symbol(require_nonaffine_sym(out));
        }

        if (n_rem.empty()) {
          // strictly less than 1 → keep as 1 / (product of leftover denom), not
          // 0
          Sym denom_red = build_prod_sym(d_rem);
          NonAffineExpr out;
          out.expr = ExprType::Div;
          out.symbols = {Sym::Const(1), denom_red};
          return Sym::Symbol(require_nonaffine_sym(out));
        }

        // reduce to Div(product(n_rem), product(d_rem))
        Sym numer_red = build_prod_sym(n_rem);
        Sym denom_red = build_prod_sym(d_rem);
        NonAffineExpr out;
        out.expr = ExprType::Div;
        out.symbols = {numer_red, denom_red};
        return Sym::Symbol(require_nonaffine_sym(out));
      }
    }

    // -------- Affine numerator ÷ symbolic denominator (term-wise cancellation
    // + rebalance + discharge) --------
    if (a.expr != ExprType::NonAffine && a.expr != ExprType::Identity) {
      // local helper: insert/merge while keeping coef sorted by symbol
      auto insert_coef_sorted = [&](auto &vec, symbol s, value_type f) {
        if (f == 0)
          return;
        auto it = std::lower_bound(
            vec.begin(), vec.end(), s,
            [](const auto &c, symbol key) { return c.sym < key; });
        if (it != vec.end() && it->sym == s) {
          auto nf = it->factor + f;
          if (nf == 0)
            vec.erase(it);
          else
            it->factor = nf;
        } else {
          vec.insert(it,
                     typename std::decay_t<decltype(vec)>::value_type{s, f});
        }
      };

      // Collect denominator multiset of symbols (Identity → [rhs], Mul → its
      // symbols)
      containers::small_vector<Sym, 2> denom_syms;
      if (b.expr == ExprType::Identity) {
        denom_syms.push_back(rhs);
      } else if (b.expr == ExprType::NonAffine) {
        const auto &bna = m_nonAffineCache.expressions[b.lhs.sym()];
        if (bna.expr != ExprType::Mul) {
          NonAffineExpr na;
          na.expr = ExprType::Div;
          na.symbols = {lhs, rhs};
          return Sym::Symbol(require_nonaffine_sym(na));
        }
        denom_syms = bna.symbols; // symbols only (by your invariant)
      } else {
        NonAffineExpr na;
        na.expr = ExprType::Div;
        na.symbols = {lhs, rhs};
        return Sym::Symbol(require_nonaffine_sym(na));
      }

      // Split numerator: Q (quotient) and R (residual that isn't divisible by
      // denom multisets)
      AffineExpr Q, R;
      Q.constant = 0;
      R.constant = a.affine.constant;
      value_type pure_den_cancels =
          0; // count of pure 'rhs' cancellations into Q.constant

      for (auto const &c : a.affine.coef) {
        const auto &term = m_expressions[c.sym];

        containers::small_vector<Sym, 2> term_syms;
        if (term.expr == ExprType::Identity) {
          term_syms.push_back(Sym::Symbol(c.sym));
        } else if (term.expr == ExprType::NonAffine) {
          const auto &tna = m_nonAffineCache.expressions[term.lhs.sym()];
          if (tna.expr != ExprType::Mul) {
            insert_coef_sorted(R.coef, c.sym, c.factor);
            continue;
          }
          term_syms = tna.symbols;
        } else {
          insert_coef_sorted(R.coef, c.sym, c.factor);
          continue;
        }

        const bool divisible = includes_multiset(
            std::span<const Sym>(term_syms.data(), term_syms.size()),
            std::span<const Sym>(denom_syms.data(), denom_syms.size()));

        if (divisible) {
          auto q_syms = multiset_diff(
              std::span<const Sym>(term_syms.data(), term_syms.size()),
              std::span<const Sym>(denom_syms.data(), denom_syms.size()));

          Sym qsym = rebuild_product_sum(
              std::span<const Sym>(q_syms.data(), q_syms.size()), lhs, rhs,
              /*dno*/ false);

          if (qsym.isConstant()) {
            Q.constant +=
                c.factor * qsym.constant(); // adds c.factor when qsym==1
            pure_den_cancels += c.factor * qsym.constant();
          } else {
            insert_coef_sorted(Q.coef, qsym.sym(), c.factor);
          }
        } else {
          insert_coef_sorted(R.coef, c.sym, c.factor);
        }
      }

      // Rebalance: if residual has negative constant and we cancelled at least
      // one pure denom, move one denom unit from Q.constant into R (R := rhs -
      // t) to enable discharge.
      if (R.coef.empty() && R.constant < 0 && pure_den_cancels > 0) {
        Q.constant -= 1;
        insert_coef_sorted(R.coef, rhs.sym(),
                           value_type{1}); // works for composite denoms too
      }

      // If nothing residual, return Q.
      if (R.coef.empty() && R.constant == 0) {
        return require_affine_sym(ExprType::Div, lhs, rhs, Q, dno);
      }

      // Discharge (den - t)/den == 0  (single denom symbol, possibly composite
      // product)
      if (residual_is_sub_rhs_t(R, rhs.sym())) {
        return require_affine_sym(ExprType::Div, lhs, rhs, Q, dno);
      }

      // Otherwise: Q + 1 * Div(R, rhs)
      Sym Rsym = require_affine_sym(ExprType::Div, lhs, rhs, R, /*dno*/ false);
      NonAffineExpr dna;
      dna.expr = ExprType::Div;
      dna.symbols = {Rsym, rhs};
      Sym divR = Sym::Symbol(require_nonaffine_sym(dna));

      AffineExpr OUT = Q;
      insert_coef_sorted(OUT.coef, divR.sym(), value_type{1});
      return require_affine_sym(ExprType::Div, lhs, rhs, OUT, dno);
    }

    // Fallback: leave as non-affine division
    NonAffineExpr nonaffine;
    nonaffine.expr = ExprType::Div;
    nonaffine.symbols = {lhs, rhs};
    return Sym::Symbol(require_nonaffine_sym(nonaffine));
  }

  Sym nonaffine_mod(Sym lhs, Sym rhs, bool dno) {
    // Trivial: 0 % Y == 0
    if (lhs.isConstant() && lhs.constant() == 0) {
      return require_const_sym(ExprType::Mod, lhs, rhs, 0, dno);
    }

    // ---- constant modulus ----
    if (rhs.isConstant()) {
      const value_type n = rhs.constant();
      assert(n > 0);
      if (n == 1) {
        return require_const_sym(ExprType::Mod, lhs, rhs, 0, dno);
      }

      // (X % m) % n  ->  X % gcd(m, n)   (only when m,n are constants)
      if (lhs.isSymbolic()) {
        const auto &le = m_expressions[lhs.sym()];
        if (le.expr == ExprType::NonAffine) {
          const auto &lna = m_nonAffineCache.expressions[le.lhs.sym()];
          if (lna.expr == ExprType::Mod && lna.symbols[1].isConstant()) {
            value_type m = lna.symbols[1].constant();
            value_type g = std::gcd(m, n);
            NonAffineExpr out;
            out.expr = ExprType::Mod;
            out.symbols = {lna.symbols[0], Sym::Const(g)}; // keep order!
            return Sym::Symbol(require_nonaffine_sym(out));
          }
        }
      }

      // Fallback: keep as non-affine mod (ordered pair)
      NonAffineExpr out;
      out.expr = ExprType::Mod;
      out.symbols = {lhs, rhs};
      return Sym::Symbol(require_nonaffine_sym(out));
    }

    // ---- symbolic modulus ----
    // Idempotence only when same modulus symbol: (X % Y) % Y == X % Y
    if (lhs.isSymbolic()) {
      const auto &le = m_expressions[lhs.sym()];
      if (le.expr == ExprType::NonAffine) {
        const auto &lna = m_nonAffineCache.expressions[le.lhs.sym()];
        if (lna.expr == ExprType::Mod && lna.symbols[1].isSymbolic() &&
            lna.symbols[1].sym() == rhs.sym()) {
          return lhs; // (X % A) % A -> X % A
        }
      }
    }

    // A % A == 0
    if (lhs.isSymbolic() && rhs.isSymbolic() && lhs.sym() == rhs.sym()) {
      return require_const_sym(ExprType::Mod, lhs, rhs, 0, dno);
    }

    // Product divisibility: (product contains all rhs factors) → 0
    auto as_mul_syms =
        [&](Sym s,
            const Expr &e) -> std::optional<containers::small_vector<Sym, 2>> {
      if (e.expr == ExprType::Identity) {
        return containers::small_vector<Sym, 2>{s};
      }
      if (e.expr == ExprType::NonAffine) {
        const auto &na = m_nonAffineCache.expressions[e.lhs.sym()];
        if (na.expr == ExprType::Mul)
          return na.symbols; // symbols only by your invariant
      }
      return std::nullopt;
    };
    if (lhs.isSymbolic() && rhs.isSymbolic()) {
      const auto &le = m_expressions[lhs.sym()];
      const auto &re = m_expressions[rhs.sym()];
      if (auto N = as_mul_syms(lhs, le)) {
        if (auto D = as_mul_syms(rhs, re)) {
          if (includes_multiset(std::span<const Sym>(N->data(), N->size()),
                                std::span<const Sym>(D->data(), D->size()))) {
            return require_const_sym(ExprType::Mod, lhs, rhs, 0, dno);
          }
        }
      }
    }

    // Fallback: keep as non-affine Mod node (ORDERED; DO NOT SORT)
    NonAffineExpr out;
    out.expr = ExprType::Mod;
    out.symbols = {lhs, rhs};
    return Sym::Symbol(require_nonaffine_sym(out));
  }

  symbol require_nonaffine_sym(const NonAffineExpr &nonaffine) {
    assert(nonaffine.expr != ExprType::Sub);
    assert(nonaffine.expr != ExprType::Add);
    symbol nonAffineSym = m_nonAffineCache.expressions.size();
    m_nonAffineCache.expressions.push_back(nonaffine);
    NonAffineExprKey key{nonAffineSym};
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

  struct SymHash {
    std::size_t operator()(const Sym &sym) const {
      std::hash<std::uint64_t> hasher;
      std::hash<std::int64_t> ihasher;
      std::size_t hash = 0x1987231298731212;
      if (sym.isConstant()) {
        hash ^= 0x178293 + (hash << 6) + (hash >> 2);
        hash ^=
            ihasher(sym.constant()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      } else {
        hash ^= 0x817432 + (hash << 6) + (hash >> 2);
        hash ^= hasher(sym.sym()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      }
      return hash;
    }
  };

  struct ModExpr {
    AffineExpr affine; // affine in mod m
  };
  struct ModSolver {
    std::vector<ModExpr> expressions;
  };
  using ModSolverHandle = std::unique_ptr<ModSolver>;
  using ModSolverCache =
      std::unordered_map<Sym, std::unique_ptr<ModSolver>, SymHash>;

  std::optional<value_type> modsolve_resume(symbol lhs, Sym rhs) {
    if (rhs.isSymbolic()) {
      return std::nullopt;
    }
    auto it = m_modSolverCache.find(rhs);
    if (it == m_modSolverCache.end()) {
      if (rhs.isSymbolic()) {
        fmt::println("modsolve[{}] does not exist yet creating", rhs.sym());
      } else {
        fmt::println("modsolve {}  does not exist yet creating",
                     rhs.constant());
      }
      auto solver = std::make_unique<ModSolver>();
      std::optional<value_type> mod = modsolve_resume_solver(solver, lhs, rhs);
      m_modSolverCache.insert(it, std::make_pair(rhs, std::move(solver)));
      return mod;
    } else {
      fmt::println("Reusing modsolver");
      auto &solver = it->second;
      return modsolve_resume_solver(solver, lhs, rhs);
    }
  }

  std::optional<value_type> modsolve_resume_solver(ModSolverHandle &solver,
                                                   symbol lhs, Sym rhs) {
    if (rhs.isConstant()) {
      const value_type m = rhs.constant();
      while (solver->expressions.size() <= lhs) {
        symbol s = solver->expressions.size();
        assert(s < m_expressions.size());

        auto expr = m_expressions[s];
        // 1. Take the complete affine expression and check if it collapses
        // under mod rhs.
        std::optional<AffineExpr> affine = affine_mod(expr.affine, m);
        if (affine.has_value() && affine->isPureConstant()) {
          ModExpr modexpr;
          modexpr.affine = *affine;
          solver->expressions.push_back(modexpr);
          continue;
        }

        // 2. Derive a affine expression mod rhs by switching over expr.expr
        std::optional<ModExpr> modexpr;

        switch (expr.expr) {
        case ExprType::Identity:
          break;
        case ExprType::NonAffine: {
          auto nonaffine = m_nonAffineCache.expressions[expr.lhs.sym()];
          switch (nonaffine.expr) {
          case ExprType::Identity:
          case ExprType::NonAffine:
          case ExprType::Add:
          case ExprType::Sub:
            throw std::runtime_error(
                "Invalid state. ID, NoAffine, Add or Sub are not valid "
                "nonaffine expression types");
            break;
          case ExprType::Div:
            modexpr = modsolve_div(solver, m, nonaffine.symbols[0],
                                   nonaffine.symbols[1]);
            break;
          case ExprType::Mod:
            modexpr = modsolve_mod(solver, m, nonaffine.symbols[0],
                                   nonaffine.symbols[1]);
            break;
          case ExprType::Mul:
            // always bail.
            break;
          }
          break;
        }
        case ExprType::Div: {
          modexpr = modsolve_div(solver, m, expr.lhs, expr.rhs);
          break;
        }
        case ExprType::Mod: {
          modexpr = modsolve_mod(solver, m, expr.lhs, expr.rhs);
          break;
        }
        case ExprType::Sub: {
          modexpr = modsolve_sub(solver, m, expr.lhs, expr.rhs);
          break;
        }
        case ExprType::Mul: {
          modexpr = modsolve_mul(solver, m, expr.lhs, expr.rhs);
          break;
        }
        case ExprType::Add: {
          modexpr = modsolve_add(solver, m, expr.lhs, expr.rhs);
          break;
        }
        }
        if (modexpr.has_value()) {
          solver->expressions.push_back(*modexpr);
        } else {
          ModExpr modexpr;
          modexpr.affine.coef.emplace_back(s, 1);
          solver->expressions.push_back(modexpr);
        }
        assert(solver->expressions.size() == s + 1);
      }
    }
    const auto &me = solver->expressions[lhs];
    if (me.affine.isPureConstant())
      return me.affine.constant;
    return std::nullopt;
  }

  static value_type emod(value_type lhs, value_type rhs) {
    assert(rhs > 0);
    value_type mod = lhs % rhs;
    if (mod < 0) {
      mod += rhs;
    }
    return mod;
  }

  static void emod_affine(AffineExpr &expr, value_type m) {
    auto it = expr.coef.begin();
    while (it != expr.coef.end()) {
      value_type factor = emod(it->factor, m);
      if (factor == 0) {
        it = expr.coef.erase(it);
        continue;
      }
      it->factor = factor;
      ++it;
    }
    expr.constant = emod(expr.constant, m);
  }

  std::optional<ModExpr> modsolve_add(const ModSolverHandle &solver,
                                      value_type m, Sym lhs, Sym rhs) {
    assert(m > 0);
    if (lhs.isConstant() && rhs.isConstant()) {
      ModExpr modexpr;
      modexpr.affine.constant = emod(lhs.constant() + rhs.constant(), m);
      return modexpr;
    }
    AffineExpr a;
    AffineExpr b;
    if (lhs.isConstant() && rhs.isSymbolic()) {
      a.constant = lhs.constant();
      b = solver->expressions[rhs.sym()].affine;
    } else if (rhs.isConstant() && lhs.isSymbolic()) {
      a = solver->expressions[lhs.sym()].affine;
      b.constant = rhs.constant();
    } else {
      a = solver->expressions[lhs.sym()].affine;
      b = solver->expressions[rhs.sym()].affine;
    }
    ModExpr modexpr;
    modexpr.affine = affine_add(a, b);
    emod_affine(modexpr.affine, m);
    return modexpr;
  }

  std::optional<ModExpr> modsolve_sub(const ModSolverHandle &solver,
                                      value_type m, Sym lhs, Sym rhs) {
    assert(m > 0);
    if (lhs.isConstant() && rhs.isConstant()) {
      ModExpr modexpr;
      modexpr.affine.constant = emod(lhs.constant() - rhs.constant(), m);
      return modexpr;
    }
    AffineExpr a;
    AffineExpr b;
    if (lhs.isConstant() && rhs.isSymbolic()) {
      a.constant = lhs.constant();
      b = solver->expressions[rhs.sym()].affine;
    } else if (rhs.isConstant() && lhs.isSymbolic()) {
      a = solver->expressions[lhs.sym()].affine;
      b.constant = rhs.constant();
    } else {
      a = solver->expressions[lhs.sym()].affine;
      b = solver->expressions[rhs.sym()].affine;
    }
    ModExpr modexpr;
    b = affine_mul(b, -1);
    modexpr.affine = affine_add(a, b);
    emod_affine(modexpr.affine, m);
    return modexpr;
  }

  std::optional<ModExpr> modsolve_mul(const ModSolverHandle &solver,
                                      value_type m, Sym lhs, Sym rhs) {
    assert(m > 0);
    if (lhs.isConstant() && rhs.isConstant()) {
      ModExpr modexpr;
      modexpr.affine.constant = emod(lhs.constant() * rhs.constant(), m);
      return modexpr;
    }
    AffineExpr a;
    AffineExpr b;
    if (lhs.isConstant() && rhs.isSymbolic()) {
      a.constant = lhs.constant();
      b = solver->expressions[rhs.sym()].affine;
    } else if (rhs.isConstant() && lhs.isSymbolic()) {
      a = solver->expressions[lhs.sym()].affine;
      b.constant = rhs.constant();
    } else {
      return std::nullopt;
    }
    ModExpr modexpr;
    auto affine = affine_mul(a, b);
    assert(affine.has_value());
    modexpr.affine = *affine;
    emod_affine(modexpr.affine, m);
    return modexpr;
  }

  std::optional<ModExpr> modsolve_div(const ModSolverHandle &solver,
                                      value_type m, Sym lhs, Sym rhs) {
    assert(m > 0);
    if (lhs.isConstant() && rhs.isConstant()) {
      ModExpr modexpr;
      assert(rhs.constant() > 0);
      modexpr.affine.constant = emod(lhs.constant() / rhs.constant(), m);
      return modexpr;
    }
    AffineExpr a;
    AffineExpr b;
    if (lhs.isConstant() && rhs.isSymbolic()) {
      a.constant = lhs.constant();
      b = solver->expressions[rhs.sym()].affine;
    } else if (rhs.isConstant() && lhs.isSymbolic()) {
      a = solver->expressions[lhs.sym()].affine;
      b.constant = rhs.constant();
    } else {
      return std::nullopt;
    }
    if (auto affine = affine_div(a, b)) {
      ModExpr modexpr;
      modexpr.affine = *affine;
      emod_affine(modexpr.affine, m);
      return modexpr;
    }
    return std::nullopt;
  }

  std::optional<ModExpr> modsolve_mod(const ModSolverHandle &solver,
                                      value_type m, Sym lhs, Sym rhs) {
    assert(m > 0);
    if (lhs.isConstant() && rhs.isConstant()) {
      ModExpr modexpr;
      assert(rhs.constant() > 0);
      modexpr.affine.constant = emod(emod(lhs.constant(), rhs.constant()), m);
      return modexpr;
    }
    AffineExpr a;
    AffineExpr b;
    if (lhs.isConstant() && rhs.isSymbolic()) {
      a.constant = lhs.constant();
      b = solver->expressions[rhs.sym()].affine;
    } else if (rhs.isConstant() && lhs.isSymbolic()) {
      a = solver->expressions[lhs.sym()].affine;
      if (rhs.constant() % m == 0) {
        return solver->expressions[lhs.sym()];
      }
      b.constant = rhs.constant();
    } else {
      a = solver->expressions[lhs.sym()].affine;
      b = solver->expressions[rhs.sym()].affine;
    }
    // very rarely successful.
    if (auto affine = affine_mod(a, b)) {
      ModExpr modexpr;
      modexpr.affine = *affine;
      emod_affine(modexpr.affine, m);
      return modexpr;
    } else if (lhs.isSymbolic()) {
      if (auto constant = modsolve_resume(lhs.sym(), rhs)) {
        ModExpr modexpr;
        modexpr.affine.constant = emod(*constant, m);
        return modexpr;
      }
    }
    return std::nullopt;
  }

  std::vector<Expr> m_expressions;
  AffineExprCache m_affineCache;
  NonAffineExprCache m_nonAffineCache;
  ModSolverCache m_modSolverCache;
};

} // namespace vkcnn
