#pragma once

#include "vkcnn/common/symbolic/Sym.hpp"
#include <cassert>
#include <complex.h>
#include <concepts>
#include <fmt/base.h>
#include <fmt/format.h>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "./SymGraph_types.inl"

namespace vkcnn {

class SymGraph {
private:
  using AffineCoef = symbolic::details::AffineCoef;
  using AffineExpr = symbolic::details::AffineExpr;
  using Expr = symbolic::details::Expr;
  using ExprType = symbolic::details::ExprType;
  using NonAffineExpr = symbolic::details::NonAffineExpr;
  using ModExpr = symbolic::details::ModExpr;
  using ModSolverHandle = symbolic::details::ModSolverHandle;

public:
  using symbol = symbolic::details::symbol;
  using value_type = symbolic::details::value_type;

  SymGraph() = default;
  SymGraph(const SymGraph &) = delete;
  SymGraph &operator=(const SymGraph &) = delete;

public:
  Sym var();

  template <typename L, typename R>
    requires(std::same_as<L, Sym> || std::is_integral_v<L>) &&
            (std::same_as<R, Sym> || std::is_integral_v<R>)
  Sym add(L lhs, R rhs, bool dno = false);

  template <typename X, typename Y, typename Z>
    requires(std::same_as<X, Sym> || std::is_integral_v<X>) &&
            (std::same_as<Y, Sym> || std::is_integral_v<Y>) &&
            (std::same_as<Z, Sym> || std::is_integral_v<Z>)
  Sym add(X x, Y y, Z z, bool dno = false);

  template <typename X, typename Y, typename Z, typename W>
    requires(std::same_as<X, Sym> || std::is_integral_v<X>) &&
            (std::same_as<Y, Sym> || std::is_integral_v<Y>) &&
            (std::same_as<Z, Sym> || std::is_integral_v<Z>) &&
            (std::same_as<W, Sym> || std::is_integral_v<W>)
  Sym add(X x, Y y, Z z, W w, bool dno = false);

  template <typename L, typename R>
    requires(std::same_as<L, Sym> || std::is_integral_v<L>) &&
            (std::same_as<R, Sym> || std::is_integral_v<R>)
  Sym sub(L lhs, R rhs, bool dno = false);

  template <typename L, typename R>
    requires(std::same_as<L, Sym> || std::is_integral_v<L>) &&
            (std::same_as<R, Sym> || std::is_integral_v<R>)
  Sym mul(L lhs, R rhs, bool dno = false);

  template <typename X, typename Y, typename Z>
    requires(std::same_as<X, Sym> || std::is_integral_v<X>) &&
            (std::same_as<Y, Sym> || std::is_integral_v<Y>) &&
            (std::same_as<Z, Sym> || std::is_integral_v<Z>)
  Sym mul(X x, Y y, Z z, bool dno = false);

  template <typename X, typename Y, typename Z, typename W>
    requires(std::same_as<X, Sym> || std::is_integral_v<X>) &&
            (std::same_as<Y, Sym> || std::is_integral_v<Y>) &&
            (std::same_as<Z, Sym> || std::is_integral_v<Z>) &&
            (std::same_as<W, Sym> || std::is_integral_v<W>)
  Sym mul(X x, Y y, Z z, W w, bool dno = false);

  template <typename L>
    requires(std::same_as<L, Sym> || std::is_integral_v<L>)
  Sym pow(L lhs, value_type rhs, bool dno = false);

  template <typename L, typename R>
    requires(std::same_as<L, Sym> || std::is_integral_v<L>) &&
            (std::same_as<R, Sym> || std::is_integral_v<R>)
  Sym div(L lhs, R rhs, bool dno = false);

  template <typename L, typename R>
    requires(std::same_as<L, Sym> || std::is_integral_v<L>) &&
            (std::same_as<R, Sym> || std::is_integral_v<R>)
  Sym cdiv(L lhs, R rhs, bool dno = false);

  template <typename L, typename R>
    requires(std::same_as<L, Sym> || std::is_integral_v<L>) &&
            (std::same_as<R, Sym> || std::is_integral_v<R>)
  Sym mod(L lhs, R rhs, bool dno = false);

  template <typename L, typename R>
    requires(std::same_as<L, Sym> || std::is_integral_v<L>) &&
            (std::same_as<R, Sym> || std::is_integral_v<R>)
  Sym alignUp(L sym, R alignment, bool dno = false);

  template <typename L, typename R>
    requires(std::same_as<L, Sym> || std::is_integral_v<L>) &&
            (std::same_as<R, Sym> || std::is_integral_v<R>)
  Sym alignDown(L sym, R alignment, bool dno = false);

  template <typename E, typename K, typename P, typename S,
            typename D = value_type>
    requires(std::same_as<E, Sym> || std::is_integral_v<E>) &&
            (std::same_as<K, Sym> || std::is_integral_v<K>) &&
            (std::same_as<P, Sym> || std::is_integral_v<P>) &&
            (std::same_as<S, Sym> || std::is_integral_v<S>) &&
            (std::same_as<D, Sym> || std::is_integral_v<D>)
  Sym pool(E extent, K kernelSize, P padding, S stride,
           D dilation = value_type(1), bool dno = false);

  template <typename E, typename K, typename P, typename S, typename D>
    requires(std::same_as<E, Sym> || std::is_integral_v<E>) &&
            (std::same_as<K, Sym> || std::is_integral_v<K>) &&
            (std::same_as<P, Sym> || std::is_integral_v<P>) &&
            (std::same_as<S, Sym> || std::is_integral_v<S>) &&
            (std::same_as<D, Sym> || std::is_integral_v<D>)
  Sym cpool(E extent, K kernelSize, P padding, S stride, D dilation = 1,
            bool dno = false);

  template <typename L, typename R>
    requires(std::same_as<L, Sym> || std::is_integral_v<L>) &&
            (std::same_as<R, Sym> || std::is_integral_v<R>)
  Sym min(L lhs, R rhs);

  template <typename L, typename R>
    requires(std::same_as<L, Sym> || std::is_integral_v<L>) &&
            (std::same_as<R, Sym> || std::is_integral_v<R>)
  Sym max(L lhs, R rhs);

  Sym abs(Sym sym) {
    throw std::runtime_error("SymGraph::abs is not implemented");
  }

  Sym neg(Sym sym) {
    throw std::runtime_error(
        "SymGraph::neg is not implemented. Requires "
        "testing that negative values are acceptable within the graph!");
    return sub(0, sym);
  }

  Sym resolve(value_type v) const;
  Sym resolve(Sym sym) const;
  void debugDump() const;

private:
  // Affine symbol cache.
  auto next_sym() -> symbol;
  auto create_variable(ExprType type) -> symbol;
  auto require_affine_sym(ExprType type, Sym lhs, Sym rhs,
                          const AffineExpr &affine, bool dno) -> Sym;
  auto require_const_sym(value_type constant, bool dno) -> Sym;
  auto find_sym_of_affine(const AffineExpr &expr) -> std::optional<symbol>;
  auto try_construct_affine_sym(const AffineExpr &expr, symbol hint,
                                std::size_t depth, bool dno)
      -> std::optional<Sym>;
  auto construct_affine_sym(AffineExpr expr, bool dno,
                            std::optional<symbol> hint = std::nullopt) -> Sym;

  // Expression handlers.
  Sym add_xx(Sym lhs, Sym rhs, bool dno = true);
  Sym add_ss(symbol lhs, symbol rhs, bool dno);
  Sym add_sc(symbol lhs, value_type rhs, bool dno);
  Sym add_cc(value_type lhs, value_type rhs, bool dno);

  Sym sub_xx(Sym lhs, Sym rhs, bool dno = true);
  Sym sub_ss(symbol lhs, symbol rhs, bool dno);
  Sym sub_sc(symbol lhs, value_type rhs, bool dno);
  Sym sub_cs(value_type lhs, symbol rhs, bool dno);
  Sym sub_cc(value_type lhs, value_type rhs, bool dno);

  Sym mul_xx(const Sym lhs, const Sym rhs, const bool dno = true);
  Sym mul_ss(symbol lhs, symbol rhs, bool dno);
  Sym mul_sc(symbol lhs, value_type rhs, bool dno);
  Sym mul_cc(value_type lhs, value_type rhs, bool dno);

  Sym div_xx(Sym lhs, Sym rhs, bool dno);
  Sym div_ss(symbol lhs, symbol rhs, bool dno);
  Sym div_sc(symbol lhs, value_type rhs, bool dno);
  Sym div_cs(value_type lhs, symbol rhs, bool dno);
  Sym div_cc(value_type lhs, value_type rhs, bool dno);

  Sym mod_xx(Sym lhs, Sym rhs, bool dno);
  Sym mod_ss(symbol lhs, symbol rhs, bool dno);
  Sym mod_sc(symbol lhs, value_type rhs, bool dno);
  Sym mod_cs(value_type lhs, symbol rhs, bool dno);
  Sym mod_cc(value_type lhs, value_type rhs, bool dno);

  Sym min_xx(Sym lhs, Sym rhs, bool dno);
  Sym max_xx(Sym lhs, Sym rhs, bool dno);

  // Affine Solvers
  static void affine_add_sym(AffineExpr &lhs, symbol s, value_type factor);
  static auto affine_add(const AffineExpr &lhs, const AffineExpr &rhs)
      -> AffineExpr;
  static void affine_mul_add_acc(AffineExpr &lhs, const AffineExpr &rhs,
                                 value_type factor);
  static void affine_add_acc(AffineExpr &lhs, const AffineExpr &rhs);
  static auto affine_sub(const AffineExpr &lhs, const AffineExpr &rhs)
      -> AffineExpr;
  static auto affine_mul(const AffineExpr &lhs, const value_type &rhs)
      -> AffineExpr;
  static void affine_mul_acc(AffineExpr &lhs, const value_type &rhs);
  static auto affine_mul(const AffineExpr &lhs, const AffineExpr &rhs)
      -> std::optional<AffineExpr>;

  auto affine_div(const AffineExpr &lhs, value_type rhs)
      -> std::optional<AffineExpr>;
  auto affine_div(const AffineExpr &lhs, const AffineExpr &rhs)
      -> std::optional<AffineExpr>;
  auto affine_mod(const AffineExpr &lhs, const value_type &rhs)
      -> std::optional<value_type>;
  auto affine_mod(const AffineExpr &lhs, const AffineExpr &rhs)
      -> std::optional<AffineExpr>;

  // Non-Affine Solver
  auto require_nonaffine_sym(const NonAffineExpr &nonaffine) -> symbol;

  Sym nonaffine_mul(symbol lhs, symbol rhs, bool dno);
  Sym nonaffine_div(Sym lhs, Sym rhs, bool dno);
  Sym nonaffine_mod(Sym lhs, Sym rhs, bool dno);

  // Mod Solver
  auto require_modsolver(Sym sym) -> const ModSolverHandle &;
  auto modsolve_resume(symbol lhs, Sym m) -> std::optional<value_type>;
  auto modsolve_reduce_symbol_mod_m(symbol sym, const Sym mod)
      -> const ModExpr &;
  auto modsolve_reduce_affine_mod_m(const AffineExpr &affine, const Sym msym)
      -> std::optional<ModExpr>;
  auto modsolve_resume_solver(const ModSolverHandle &solver, symbol lhs,
                              Sym rhs) -> std::optional<value_type>;
  auto modsolve_reverse_peel(const AffineExpr &expr, std::size_t m)
      -> std::optional<value_type>;
  auto modsolve_mul_only_exact(const AffineExpr &lhs, Sym rhs)
      -> std::pair<AffineExpr, AffineExpr>;
  auto modsolve_peel_by_d(const AffineExpr &lhs, value_type d)
      -> std::pair<AffineExpr, AffineExpr>;
  auto modsolve_affine_add(value_type m, const AffineExpr &lhs,
                           const AffineExpr &rhs) -> ModExpr;
  void modsolve_affine_add_acc(value_type m, ModExpr &lhs,
                               const AffineExpr &rhs);
  void modsolve_affine_mul_add_acc(value_type m, ModExpr &lhs,
                                   const AffineExpr &rhs, value_type v);

  auto modsolve_add(const ModSolverHandle &solver, value_type m, Sym lhs,
                    Sym rhs) -> ModExpr;
  auto modsolve_sub(const ModSolverHandle &solver, value_type m, Sym lhs,
                    Sym rhs) -> ModExpr;
  auto modsolve_mul(const ModSolverHandle &solver, value_type m, Sym lhs,
                    Sym rhs) -> std::optional<ModExpr>;
  auto modsolve_div(value_type m, Sym lhs, Sym rhs) -> std::optional<ModExpr>;
  auto modsolve_mod(const ModSolverHandle &solver, value_type m, Sym lhs,
                    Sym rhs) -> std::optional<ModExpr>;

  // Helpers:
  static auto floordivmod(value_type a, value_type b)
      -> std::pair<value_type, value_type>;
  static bool includes_multiset(std::span<const Sym> A, std::span<const Sym> B);
  static auto multiset_diff(std::span<const Sym> A, std::span<const Sym> B)
      -> NonAffineExpr::SmallVector;
  static value_type emod(value_type lhs, value_type rhs);

  static void emod_affine(AffineExpr &expr, value_type m);

  std::vector<Expr> m_expressions;
  symbolic::details::AffineExprCache m_affineCache;
  symbolic::details::NonAffineExprCache m_nonAffineCache;
  symbolic::details::ModSolverCache m_modSolverCache;
};

} // namespace vkcnn

// Templated functions definitions.
#include "./SymGraph_api_templates.inl"
