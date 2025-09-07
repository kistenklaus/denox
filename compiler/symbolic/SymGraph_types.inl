#include "memory/container/hashmap.hpp"
#include "memory/container/small_vector.hpp"
#include "memory/container/vector.hpp"
#include "memory/container/shared_ptr.hpp"
#include "symbolic/Sym.hpp"
#include <cstdint>

namespace denox::compiler::symbolic::details {

using symbol = Sym::symbol;
using value_type = Sym::value_type;

struct AffineCoef {
  symbol sym;
  value_type factor;
};

struct AffineExpr {
  using SmallVector = denox::memory::small_vector<AffineCoef, 2>;
  // NOTE: The coef are always sorted by sym (Invariant)
  SmallVector coef;
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
    denox::memory::hash_map<AffineExpr, symbol, AffineExprHash, AffineExprComp>;

enum class ExprType {
  Identity,  // (a = 0, b = 0)=
  NonAffine, // (a -> nonaffine cache)
             //
  Div,       // a / b
  Mod,       // a % b
  Sub,       // a - b
  Mul,       // a * b
  Add,       // a + b
  Min,       // min(a,b)
  Max,       // max(a,b)
  Const,
};

struct Expr {
  ExprType expr = ExprType::Identity;
  AffineExpr affine;
  Sym lhs = Sym::Const(0);
  Sym rhs = Sym::Const(0);
};

struct NonAffineExpr {
  using SmallVector = denox::memory::small_vector<Sym, 2>;
  ExprType expr;
  // NOTE: symbols might be a multiset for example 4,X,X,Y is valid!
  // For associative ops symbols are sorted (constants asc, symbols asc)
  SmallVector symbols;
  symbol sym = symbol(-1);
};

struct NonAffineExprKey {
  symbol nonaffinesym;
};

using NonAffineExprArray = denox::memory::vector<NonAffineExpr>;

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
          bhasher(sym.isConstant()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      hash ^=
          hasher(sym.isConstant() ? static_cast<std::uint64_t>(sym.constant())
                                  : sym.sym()) +
          0x9e3779b9 + (hash << 6) + (hash >> 2);
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
  using HashMap = denox::memory::hash_map<NonAffineExprKey, symbol,
                                          NonAffineExprHash, NonAffineExprComp>;

  NonAffineExprCache(const NonAffineExprCache &o)
      : expressions(o.expressions), cache(0, NonAffineExprHash(&expressions),
                                          NonAffineExprComp(&expressions)) {}

  NonAffineExprCache &operator=(const NonAffineExprCache &o) {
    if (this == &o) {
      return *this;
    }
    expressions = o.expressions;
    cache = HashMap(0, NonAffineExprHash(&expressions),
                    NonAffineExprComp(&expressions));
    for (const auto &k : o.cache) {
      cache.insert(k);
    }
    return *this;
  }

  NonAffineExprCache()
      : expressions(), cache(0, NonAffineExprHash(&expressions),
                             NonAffineExprComp(&expressions)) {}

  NonAffineExprArray expressions;
  HashMap cache;
};

struct SymHash {
  std::size_t operator()(const Sym &sym) const {
    std::hash<std::uint64_t> hasher;
    std::hash<std::int64_t> ihasher;
    std::size_t hash = 0x1987231298731212;
    if (sym.isConstant()) {
      hash ^= 0x178293 + (hash << 6) + (hash >> 2);
      hash ^= ihasher(sym.constant()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
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
  denox::memory::vector<ModExpr> expressions;
};
using ModSolverHandle = denox::memory::shared_ptr<ModSolver>;
using ModSolverCache = denox::memory::hash_map<Sym, ModSolverHandle, SymHash>;

} // namespace denox::compiler::symbolic::details
