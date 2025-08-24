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
  Sym div(L lhs, R rhs, bool ceil = false, bool dno = true) {
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
    return div_xx(a, b, ceil, dno);
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
      case ExprType::CeilDiv:
        assert(expr.symbols.size() == 2);
        fmt::print("CeilDiv({}, {})", form(expr.symbols[0]),
                   form(expr.symbols[1]));
        break;
      case ExprType::FloorDiv:
        assert(expr.symbols.size() == 2);
        fmt::print("FloorDiv({}, {})", form(expr.symbols[0]),
                   form(expr.symbols[1]));
        break;
      case ExprType::AlignUp:
        assert(expr.symbols.size() == 2);
        fmt::print("AlignUp({}, {})", form(expr.symbols[0]),
                   form(expr.symbols[1]));
        break;
      case ExprType::Mod:
        assert(expr.symbols.size() == 2);
        fmt::print("Mod({}, {})", form(expr.symbols[0]), form(expr.symbols[1]));
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
      case ExprType::Max:
        fmt::print("Max(");
        for (std::size_t c = 0; c < expr.symbols.size(); ++c) {
          if (c != 0) {
            fmt::print(", ");
          }
          fmt::print("{}", form(expr.symbols[c]));
        }
        fmt::print(")");
        break;
      case ExprType::Min:
        fmt::print("Min(");
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
      case ExprType::CeilDiv:
        exprStr = fmt::format("{} cdiv {}", avar, bvar);
        break;
      case ExprType::FloorDiv:
        exprStr = fmt::format("{} div {}", avar, bvar);
        break;
      case ExprType::AlignUp:
        exprStr = fmt::format("alignup({}, {})", avar, bvar);
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
      case ExprType::Max:
        exprStr = fmt::format("max({}, {})", avar, bvar);
        break;
      case ExprType::Min:
        exprStr = fmt::format("min({}, {})", avar, bvar);
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
    value_type constant;

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
    CeilDiv,   // (a + b - 1) / b
    FloorDiv,  // a / b
    AlignUp,   // ((a+b-1)/b)*b   #b must be a power of 2!
    Mod,       // a % b
    Sub,       // a-b
    Mul,       // a*b*c
    Add,       // a+b+c
    Max,       // max(a,b,c)
    Min,       // min(a,b,c)
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

  Sym div_xx(Sym lhs, Sym rhs, bool ceilMode, bool dno) {
    if (lhs.isConstant() && rhs.isConstant()) {
      return div_cc(lhs.constant(), rhs.constant(), ceilMode, dno);
    } else if (lhs.isConstant()) {
      return div_cs(lhs.constant(), rhs.sym(), ceilMode, dno);
    } else if (rhs.isConstant()) {
      return div_sc(lhs.sym(), rhs.constant(), ceilMode, dno);
    } else {
      return div_ss(lhs.sym(), rhs.sym(), ceilMode, dno);
    }
  }

  Sym div_ss(symbol lhs, symbol rhs, bool ceilMode, bool dno) {
    const auto &a = m_expressions[lhs];
    const auto &b = m_expressions[rhs];
    std::optional<AffineExpr> affine = affine_div(a.affine, b.affine, ceilMode);
    if (affine.has_value()) {
      return require_affine_sym(
          ceilMode ? ExprType::CeilDiv : ExprType::FloorDiv, Sym::Symbol(lhs),
          Sym::Symbol(rhs), *affine, dno);
    } else {
      return nonaffine_div(Sym::Symbol(lhs), Sym::Symbol(rhs), ceilMode, dno);
    }
  }

  Sym div_sc(symbol lhs, value_type rhs, bool ceilMode, bool dno) {
    const auto &a = m_expressions[lhs];
    std::optional<AffineExpr> affine = affine_div(a.affine, rhs, ceilMode);
    if (affine.has_value()) {
      return require_affine_sym(
          ceilMode ? ExprType::CeilDiv : ExprType::FloorDiv, Sym::Symbol(lhs),
          Sym::Const(rhs), *affine, dno);
    } else {
      return nonaffine_div(Sym::Symbol(lhs), Sym::Const(rhs), ceilMode, dno);
    }
  }

  Sym div_cs(value_type lhs, symbol rhs, bool ceilMode, bool dno) {
    if (lhs == 0) {
      AffineExpr affine;
      affine.constant = 0;
      return require_affine_sym(ceilMode ? ExprType::CeilDiv
                                         : ExprType::FloorDiv,
                                Sym::Const(lhs), Sym::Symbol(rhs), affine, dno);
    } else {
      return nonaffine_div(Sym::Const(lhs), Sym::Symbol(rhs), ceilMode, dno);
    }
  }

  Sym div_cc(value_type lhs, value_type rhs, bool ceilMode, bool dno) {
    AffineExpr affine;
    assert(rhs > 0);
    if (ceilMode) {
      affine.constant = (lhs + rhs - 1) / rhs;
    } else {
      affine.constant = lhs / rhs;
    }
    return require_affine_sym(ceilMode ? ExprType::CeilDiv : ExprType::FloorDiv,
                              Sym::Const(lhs), Sym::Const(rhs), affine, dno);
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
      return affine;affine;
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

  std::optional<AffineExpr> affine_div(const AffineExpr &lhs, value_type rhs,
                                       bool ceilMode) {
    assert(rhs > 0);
    if (rhs == 1) {
      return lhs;
    }
    if (lhs.coef.empty()) {
      // NOTE: constant folding
      AffineExpr out;
      if (lhs.constant == 0) {
        return out;
      } else {
        if (ceilMode) {
          out.constant = (lhs.constant + rhs - 1) / rhs;
        } else {
          out.constant = lhs.constant / rhs;
        }
      }
      return out;
    }
    value_type div = rhs;
    assert(div != 0);
    bool allDivisible = true;
    if (lhs.constant % div != 0) {
      // bail.
      allDivisible = false;
    }
    for (std::size_t c = 0; c < lhs.coef.size() && allDivisible; ++c) {
      if (lhs.coef[c].factor % div != 0) {
        allDivisible = false;
      }
    }
    if (allDivisible) {
      AffineExpr out;
      out = lhs;
      for (auto &coef : out.coef) {
        coef.factor /= div;
      }
      out.constant /= div;
      return out;
    }
    return std::nullopt;
  }

  std::optional<AffineExpr> affine_div(const AffineExpr &lhs,
                                       const AffineExpr &rhs, bool ceilMode) {
    if (rhs.coef.empty()) {
      assert(rhs.constant > 0);
      return affine_div(lhs, rhs.constant, ceilMode);
    }

    if (lhs.coef.empty()) {
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
      if (ceilMode) {
        out.constant = (a.factor + b.factor - 1) / b.factor;
      } else {
        out.constant = a.factor / b.factor;
      }
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
      NonAffineExpr nonaffine;
      nonaffine.expr = ExprType::Mul;
      if (a.expr == ExprType::NonAffine) {
        const auto a_nonaffine = m_nonAffineCache.expressions[a.lhs.sym()];
        if (a_nonaffine.expr == ExprType::Mul) {
          value_type constant = 1;
          for (std::size_t c = 0; c < a_nonaffine.symbols.size(); ++c) {
            const auto sym = a_nonaffine.symbols[c];
            if (sym.isConstant()) {
              constant *= sym.constant();
            } else {
              nonaffine_add_sym_associative(nonaffine, sym);
            }
          }
          if (constant != 1) {
            nonaffine_add_sym_associative(nonaffine, Sym::Const(constant));
          }

        } else {
          nonaffine_add_sym_associative(nonaffine, Sym::Symbol(lhs));
        }
      } else {
        // lhs is identity A.
        // if rhs is identity -> simply produce AX
        if (b.expr == ExprType::Identity) {
          // rhs is identity and
          nonaffine_add_sym_associative(nonaffine, Sym::Symbol(lhs));
        } else {
          // lhs is identity and rhs is nonaffine, forward to associative case.
          std::swap(a,b);
          std::swap(lhs, rhs);
        }
      }
      if (b.expr == ExprType::NonAffine) {
        const auto &b_nonaffine = m_nonAffineCache.expressions[b.lhs.sym()];
        if (b_nonaffine.expr == ExprType::Mul) {
          value_type constant = 1;
          for (std::size_t c = 0; c < b_nonaffine.symbols.size(); ++c) {
            const auto sym = b_nonaffine.symbols[c];
            if (sym.isConstant()) {
              constant *= sym.constant();
            } else {
              nonaffine_add_sym_associative(nonaffine, sym);
            }
          }
          if (constant != 1) {
            nonaffine_add_sym_associative(nonaffine, Sym::Const(constant));
          }
        } else {
          nonaffine_add_sym_associative(nonaffine, Sym::Symbol(rhs));
        }
      } else {
        nonaffine_add_sym_associative(nonaffine, Sym::Symbol(rhs));
      }
      return Sym::Symbol(require_nonaffine_sym(nonaffine));
    }
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

  Sym nonaffine_div(Sym lhs, Sym rhs, bool ceilMode, bool dno) {
    // NOTE: either lhs or rhs are symbolics, if both are const, then they
    // would have been handled by affine_div.
    assert(lhs.isSymbolic() || rhs.isSymbolic());
    if (lhs.isConstant()) {
      if (lhs.constant() == 0) {
        assert(rhs.isSymbolic());
        // NOTE: Trivial case 0 / X == 0
        return require_const_sym(ceilMode ? ExprType::CeilDiv
                                          : ExprType::FloorDiv,
                                 lhs, rhs, 0, dno);
      } else if (lhs.constant() == 1) {
        // NOTE: This uses the fact that if rhs == 0, we would get UB so we
        // assume that it does not occure, and can optimize this case!
        return require_const_sym(ceilMode ? ExprType::CeilDiv
                                          : ExprType::FloorDiv,
                                 lhs, rhs, ceilMode ? 1 : 0, dno);
      } else {
        NonAffineExpr nonaffine;
        nonaffine.expr = ceilMode ? ExprType::CeilDiv : ExprType::FloorDiv;
        nonaffine.symbols = {lhs, rhs};
        return Sym::Symbol(require_nonaffine_sym(nonaffine));
      }
    }

    if (rhs.isConstant()) {
      assert(lhs.isSymbolic());
      const auto &a = m_expressions[lhs.sym()];
      if (a.expr == ExprType::NonAffine) {
        const auto &a_nonaffine = m_nonAffineCache.expressions[a.lhs.sym()];
        if (ceilMode) {
          if (a_nonaffine.expr == ExprType::CeilDiv) {
            if (a_nonaffine.symbols[1].isConstant()) {
              assert(a_nonaffine.symbols[0].isSymbolic());
              NonAffineExpr nonaffine;
              value_type div =
                  a_nonaffine.symbols[1].constant() * rhs.constant();
              nonaffine.expr = ExprType::CeilDiv;
              nonaffine.symbols = {a_nonaffine.symbols[0], Sym::Const(div)};
              return Sym::Symbol(require_nonaffine_sym(nonaffine));
              return Sym::Symbol(require_nonaffine_sym(nonaffine));
            } else if (a_nonaffine.symbols[0].isConstant()) {
              // NOTE: Given (c0 cdiv X) cdiv c1, if c0 < c1 we can prove that
              // c0 cdiv X < c1. We also know that c0 cdiv X != 0, because
              // that's only the case if c0 is 0, which is a trivial case that
              // would have been handeled by the affine_div. This implies that
              // (c0 cdiv X) cdiv c1 with c0 < c1 is always 1.
              value_type c0 = a_nonaffine.symbols[0].isConstant();
              value_type c1 = rhs.constant();
              if (c0 < c1) {
                return require_const_sym(ExprType::FloorDiv, lhs, rhs, 1, dno);
              }
            }
          }
        } else {
          if (a_nonaffine.expr == ExprType::FloorDiv) {
            if (a_nonaffine.symbols[1].isConstant()) {
              assert(a_nonaffine.symbols[0].isSymbolic());
              NonAffineExpr nonaffine;
              value_type div =
                  a_nonaffine.symbols[1].constant() * rhs.constant();
              nonaffine.expr = ExprType::FloorDiv;
              nonaffine.symbols = {a_nonaffine.symbols[1], Sym::Const(div)};
              return Sym::Symbol(require_nonaffine_sym(nonaffine));
            } else if (a_nonaffine.symbols[0].isConstant()) {
              // NOTE: Given (c0 div X) div c1, if c0 < c1 we can prove that c0
              // div X < c1 which implies that (c0 div X) div c1 = 0.
              value_type c0 = a_nonaffine.symbols[0].isConstant();
              value_type c1 = rhs.constant();
              if (c0 < c1) {
                return require_const_sym(ExprType::FloorDiv, lhs, rhs, 0, dno);
              }
            }
          }
        }
      }

      // NOTE: Bail
      NonAffineExpr nonaffine;
      nonaffine.expr = ceilMode ? ExprType::CeilDiv : ExprType::FloorDiv;
      nonaffine.symbols = {lhs, rhs};
      return Sym::Symbol(require_nonaffine_sym(nonaffine));
    }

    assert(lhs.isSymbolic() && rhs.isSymbolic());

    // NOTE: This case should be covered by affine_div.
    assert(lhs.sym() != rhs.sym());
    const auto &a = m_expressions[lhs.sym()];
    const auto &b = m_expressions[rhs.sym()];

    if (a.expr == ExprType::NonAffine) {
      const auto &a_nonaffine = m_nonAffineCache.expressions[a.lhs.sym()];
      if (b.expr == ExprType::NonAffine) {

        // NOTE: Case: X div Y = ABC div BC = A. This is true with ceiling or
        // flooring division, iff. X is proofably a exact multiple of Y so in
        // other words only if numerator contains all symbols of denominator.

        const auto &b_nonaffine = m_nonAffineCache.expressions[b.lhs.sym()];
        if (a_nonaffine.expr == ExprType::Mul &&
            b_nonaffine.expr == ExprType::Mul) {
          auto ai = a_nonaffine.symbols.begin();
          auto bi = b_nonaffine.symbols.begin();

          bool divisible = true;

          NonAffineExpr nonaffine;
          while (ai != a_nonaffine.symbols.end() &&
                 bi != b_nonaffine.symbols.end()) {
            assert(ai->isSymbolic());
            assert(bi->isSymbolic());
            if (ai->sym() == bi->sym()) {
              // scale symbol.
              ++ai;
              ++bi;
            } else if (ai->sym() < bi->sym()) {
              // symbol only in numerator
              nonaffine.symbols.push_back(*ai);
              ++ai;
            } else {
              // denom has factor not in numerator -> not divisible bail!
              divisible = false;
              break;
            }
          }
          while (ai != a_nonaffine.symbols.end()) {
            nonaffine.symbols.push_back(*ai++);
          }
          if (bi != b_nonaffine.symbols.end()) {
            // denom has factor not in numerator -> not divisible bail!
            divisible = false;
          }

          if (divisible) {
            // NOTE: Trivial cases like ABC / ABC, are already handled in the
            // affine domain. Because both sides are repr. as the same symbol.
            assert(!nonaffine.symbols.empty());
            if (nonaffine.symbols.size() == 1) {
              // ABC / BC = A
              return nonaffine.symbols[0];
            } else {
              // XYZW / XY = ZW
              nonaffine.expr = ExprType::Mul;
              return Sym::Symbol(require_nonaffine_sym(nonaffine));
            }
          } else {
          }
        }
      } else if (b.expr == ExprType::Identity) {

        if (a_nonaffine.expr == ExprType::Mul) {
          const symbol bsym = b.lhs.sym();
          NonAffineExpr nonaffine;
          bool cancelled = false;
          for (const auto &asym : a_nonaffine.symbols) {
            assert(asym.isSymbolic());
            if (!cancelled && asym.sym() == bsym) {
              cancelled = true;
              continue;
            }
            nonaffine.symbols.push_back(asym);
          }
          if (cancelled) {
            // NOTE: Cases like B / B, are trivial and handled in the
            // affine domain.
            assert(!nonaffine.symbols.empty());
            if (nonaffine.symbols.size() == 1) {
              // AB / B = A -> move back up into affine domain.
              return nonaffine.symbols[0];
            } else {
              nonaffine.expr = ExprType::Mul;
              return Sym::Symbol(require_nonaffine_sym(nonaffine));
            }
          }
        }
      }
    } else if (a.expr == ExprType::Identity) {
      if (b.expr == ExprType::NonAffine) {
        const symbol asym = a.lhs.sym();
        const auto &b_nonaffine = m_nonAffineCache.expressions[b.lhs.sym()];
        if (b_nonaffine.expr == ExprType::Mul) {
          bool cancelled = false;
          assert(b_nonaffine.symbols.size() >= 2);
          for (const auto &bsym : b_nonaffine.symbols) {
            assert(bsym.isSymbolic());
            if (!cancelled && bsym.sym() == asym) {
              cancelled = true;
              break;
            }
          }
          if (cancelled) {
            if (ceilMode) {
              // NOTE: A cdiv ABC = 1 unless A is 0, but if A = 0, then
              // ABC = 0, which is UB. We therefor assume that A != 0,
              // which means that A cdiv ABC = 1.
              return require_const_sym(ExprType::CeilDiv, lhs, rhs, 1, dno);
            } else {
              // NOTE: A div ABC = 0 iff. BC > 0. Because BC apears in the
              // denominator BC = 0 would imply ABC = 0, which is a division by
              // zero. Because division by zero is UB we can safely assume that
              // BC > 0.
              return require_const_sym(ExprType::FloorDiv, lhs, rhs, 0, dno);
            }
          }
        }
      }
    }

    NonAffineExpr nonaffine;
    nonaffine.expr = ceilMode ? ExprType::CeilDiv : ExprType::FloorDiv;
    nonaffine.symbols = {lhs, rhs};
    return Sym::Symbol(require_nonaffine_sym(nonaffine));
  }

  Sym nonaffine_mod(Sym lhs, Sym rhs, [[maybe_unused]] bool dno) {
    NonAffineExpr nonaffine;
    nonaffine.expr = ExprType::Mod;
    nonaffine.symbols = {lhs, rhs};
    return Sym::Symbol(require_nonaffine_sym(nonaffine));
  }

  symbol require_nonaffine_sym(const NonAffineExpr &nonaffine) {
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
      auto solver = std::make_unique<ModSolver>();
      m_modSolverCache.insert(it, std::make_pair(rhs, std::move(solver)));
      return modsolve_resume_solver(solver, lhs, rhs);
    } else {
      auto &solver = it->second;
      return modsolve_resume_solver(solver, lhs, rhs);
    }
  }

  std::optional<value_type> modsolve_resume_solver(ModSolverHandle &solver,
                                                   symbol lhs, Sym rhs) {
    if (rhs.isConstant()) {
      while (solver->expressions.size() <= lhs) {
        symbol s = solver->expressions.size();
        assert(s < m_expressions.size());

        auto expr = m_expressions[s];

        // 1. Take the complete affine expression and check if it collapses
        // under mod rhs.
        //
        // 2. Derive a affine expression mod rhs by switching over expr.expr
        //

        // mod solve.
      }
    } else {
      return std::nullopt;
    }
    const auto &expr = solver->expressions[lhs];
    if (expr.affine.isPureConstant()) {
      return expr.affine.constant;
    } else {
      return std::nullopt;
    }
  }

  std::vector<Expr> m_expressions;
  AffineExprCache m_affineCache;
  NonAffineExprCache m_nonAffineCache;
  ModSolverCache m_modSolverCache;
};

} // namespace vkcnn
