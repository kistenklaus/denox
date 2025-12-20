#pragma once

#include "denox/symbolic/Sym.hpp"
#include <cassert>

namespace denox::compiler {

struct sym {
private:
  static constexpr Sym::symbol mask63 =
      (Sym::symbol{1} << 63) - 1; // 0x7FFF...FFFF
public:
  sym() : m_isConstant(false), m_value(0) {}
  sym(Sym s)
      : m_isConstant(s.isConstant()),
        m_value(
            (m_isConstant ? static_cast<Sym::symbol>(s.constant()) : s.sym()) &
            mask63) {
    assert(!s.isConstant() || s.constant() >= 0);
  }

  bool isConstant() const { return m_isConstant; }

  bool isSymbolic() const { return !m_isConstant; }

  Sym::symbol symbol() const { return m_value; }

  Sym::value_type constant() const {
    return static_cast<Sym::value_type>(m_value);
  }

  Sym asSym() const {
    if (m_isConstant) {
      return Sym::Const(static_cast<Sym::value_type>(m_value));
    } else {
      return Sym::Symbol(m_value);
    }
  }

private:
  Sym::symbol m_isConstant : 1;
  Sym::symbol m_value : 63;
};

static_assert(sizeof(sym) == sizeof(Sym::symbol));

} // namespace denox::compiler
