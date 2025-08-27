#pragma once

#include <cassert>
#include <cstdint>
namespace vkcnn {

class SymGraph;

struct Sym {
  friend SymGraph;
  using value_type = std::int64_t;
  using symbol = std::uint64_t;

  // NOTE: These are really dangerous during development, but the interface
  // itself it great! operator bool() const { return m_isConstant; }

  bool isSymbolic() const { return !m_isConstant; }
  bool isConstant() const { return m_isConstant; }

  [[deprecated("Use Sym::constant() instead!")]]
  value_type value() const {
    assert(m_isConstant);
    return m_constant;
  }

  value_type constant() const {
    assert(m_isConstant);
    return m_constant;
  }
  // std::int64_t operator*() const { return m_constant; }

  symbol sym() const {
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

  static Sym Const(value_type v) { return Sym{v}; }
  static Sym Symbol(symbol sym) { return Sym{sym}; }

private:
  Sym() : m_isConstant(true), m_constant(0) {}
  explicit Sym(value_type v) : m_isConstant(true), m_constant(v) {}
  explicit Sym(symbol sym) : m_isConstant(false), m_sym(sym) {}

  bool m_isConstant;
  union {
    std::uint64_t m_sym;
    std::int64_t m_constant;
  };
};

} // namespace vkcnn
