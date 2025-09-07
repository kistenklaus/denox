#pragma once

#include "symbolic/Sym.hpp"
#include <cassert>

namespace denox::compiler {

struct sym {
public:
  sym(Sym sym)
      : m_isConstant(sym.isConstant()),
        m_value(m_isConstant ? static_cast<Sym::symbol>(sym.constant())
                             : sym.sym()) {
    assert(!sym.isConstant() || (sym.constant() >= 0));
  }

  bool isConstant() const { return m_isConstant; }

  bool isSymbolic() const { return !m_isConstant; }

  Sym symbol() const { return Sym::Symbol(m_value); }

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
