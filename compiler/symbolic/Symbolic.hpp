#pragma once

#include "symbolic/SymGraph.hpp"

namespace denox::compiler {

class Symbolic {
public:
  explicit Symbolic(SymGraph *graph, Sym self)
      : m_symGraph(graph), m_self(self) {
    if (self.isSymbolic()) {
      assert(graph != nullptr);
    }
  }

  friend Symbolic operator+(const Symbolic &a, const Symbolic &b) {
    assert(!a.m_self.isSymbolic() || a.m_symGraph != nullptr);
    assert(!b.m_self.isSymbolic() || b.m_symGraph != nullptr);
    if (a.m_self.isConstant() && b.m_self.isConstant()) {
      return constBinaryOp(a, b, a.m_self.constant() + b.m_self.constant());
    }
    SymGraph *symGraph = selectSymGraph(a, b);
    assert(symGraph != nullptr);
    return Symbolic(symGraph, symGraph->add(a.m_self, b.m_self));
  }

  friend Symbolic operator+(const Symbolic &a, Sym::value_type b) {
    if (a.m_self.isConstant()) {
      return constBinaryOp(a, a.m_self.constant() + b);
    }
    assert(a.m_symGraph != nullptr);
    return Symbolic(a.m_symGraph, a.m_symGraph->add(a.m_self, b));
  }

  friend Symbolic operator+(Sym::value_type a, const Symbolic &b) {
    if (b.isConstant()) {
      return constBinaryOp(b, b.m_self.constant() + a);
    }
    assert(b.m_symGraph != nullptr);
    return Symbolic(b.m_symGraph, b.m_symGraph->add(a, b.m_self));
  }

  friend Symbolic operator-(const Symbolic &a, const Symbolic &b) {
    assert(!a.m_self.isSymbolic() || a.m_symGraph != nullptr);
    assert(!b.m_self.isSymbolic() || b.m_symGraph != nullptr);
    if (a.m_self.isConstant() && b.m_self.isConstant()) {
      return constBinaryOp(a, b, a.m_self.constant() - b.m_self.constant());
    }
    SymGraph *symGraph = selectSymGraph(a, b);
    assert(symGraph != nullptr);
    return Symbolic(symGraph, symGraph->sub(a.m_self, b.m_self));
  }

  friend Symbolic operator-(const Symbolic &a, Sym::value_type b) {
    if (a.isConstant()) {
      return constBinaryOp(a, a.m_self.constant() - b);
    }
    assert(a.m_symGraph != nullptr);
    return Symbolic(a.m_symGraph, a.m_symGraph->sub(a.m_self, b));
  }

  friend Symbolic operator-(Sym::value_type a, const Symbolic &b) {
    if (b.isConstant()) {
      return constBinaryOp(b, b.m_self.constant() - a);
    }
    assert(b.m_symGraph != nullptr);
    return Symbolic(b.m_symGraph, b.m_symGraph->sub(a, b.m_self));
  }

  friend Symbolic operator*(const Symbolic &a, const Symbolic &b) {
    assert(!a.m_self.isSymbolic() || a.m_symGraph != nullptr);
    assert(!b.m_self.isSymbolic() || b.m_symGraph != nullptr);
    if (a.m_self.isConstant() && b.m_self.isConstant()) {
      return constBinaryOp(a, b, a.m_self.constant() * b.m_self.constant());
    }
    SymGraph *symGraph = selectSymGraph(a, b);
    assert(symGraph != nullptr);
    return Symbolic(symGraph, symGraph->mul(a.m_self, b.m_self));
  }

  friend Symbolic operator*(const Symbolic &a, Sym::value_type b) {
    if (a.isConstant()) {
      return constBinaryOp(a, a.m_self.constant() * b);
    }
    assert(a.m_symGraph != nullptr);
    return Symbolic(a.m_symGraph, a.m_symGraph->mul(a.m_self, b));
  }

  friend Symbolic operator*(Sym::value_type &a, const Symbolic &b) {
    if (b.isConstant()) {
      return constBinaryOp(b, b.m_self.constant() * a);
    }
    assert(b.m_symGraph != nullptr);
    return Symbolic(b.m_symGraph, b.m_symGraph->mul(a, b.m_self));
  }

  friend Symbolic operator/(const Symbolic &a, const Symbolic &b) {
    assert(!a.m_self.isSymbolic() || a.m_symGraph != nullptr);
    assert(!b.m_self.isSymbolic() || b.m_symGraph != nullptr);
    if (a.m_self.isConstant() && b.m_self.isConstant()) {
      return constBinaryOp(a, b, a.m_self.constant() / b.m_self.constant());
    }
    SymGraph *symGraph = selectSymGraph(a, b);
    assert(symGraph != nullptr);
    return Symbolic(symGraph, symGraph->div(a.m_self, b.m_self));
  }

  friend Symbolic operator/(const Symbolic &a, Sym::value_type b) {
    if (a.isConstant()) {
      return constBinaryOp(a, a.m_self.constant() / b);
    }
    assert(a.m_symGraph != nullptr);
    return Symbolic(a.m_symGraph, a.m_symGraph->div(a.m_self, b));
  }

  friend Symbolic operator/(Sym::value_type &a, const Symbolic &b) {
    if (b.isConstant()) {
      return constBinaryOp(b, b.m_self.constant() / a);
    }
    assert(b.m_symGraph != nullptr);
    return Symbolic(b.m_symGraph, b.m_symGraph->div(a, b.m_self));
  }

  friend Symbolic operator%(const Symbolic &a, const Symbolic &b) {
    assert(!a.m_self.isSymbolic() || a.m_symGraph != nullptr);
    assert(!b.m_self.isSymbolic() || b.m_symGraph != nullptr);
    if (a.m_self.isConstant() && b.m_self.isConstant()) {
      return constBinaryOp(a, b, a.m_self.constant() % b.m_self.constant());
    }
    SymGraph *symGraph = selectSymGraph(a, b);
    assert(symGraph != nullptr);
    return Symbolic(symGraph, symGraph->mod(a.m_self, b.m_self));
  }

  friend Symbolic operator%(const Symbolic &a, Sym::value_type b) {
    if (a.isConstant()) {
      return constBinaryOp(a, a.m_self.constant() % b);
    }
    assert(a.m_symGraph != nullptr);
    return Symbolic(a.m_symGraph, a.m_symGraph->mod(a.m_self, b));
  }

  friend Symbolic operator%(Sym::value_type &a, const Symbolic &b) {
    if (b.isConstant()) {
      return constBinaryOp(b, b.m_self.constant() % a);
    }
    assert(b.m_symGraph != nullptr);
    return Symbolic(b.m_symGraph, b.m_symGraph->mod(a, b.m_self));
  }

  const Sym &operator*() const { return m_self; }

  operator Sym() const { return m_self; }

  friend bool operator==(const Symbolic &a, const Symbolic &b) {
    SymGraph *symGraph = selectSymGraph(a, b);
    if (symGraph != nullptr) {
      return symGraph->resolve(a.m_self) == symGraph->resolve(b.m_self);
    } else {
      return a.m_self == b.m_self;
    }
  }

  friend bool operator==(Sym::value_type a, const Symbolic &b) {
    if (b.m_symGraph != nullptr) {
      return b.m_symGraph->resolve(a) == b.m_symGraph->resolve(b.m_self);
    } else {
      return Sym::Const(a) == b.m_self;
    }
  }

  friend bool operator==(const Symbolic &a, Sym::value_type b) {
    if (a.m_symGraph != nullptr) {
      return a.m_symGraph->resolve(a.m_self) == a.m_symGraph->resolve(b);
    } else {
      return a.m_self == Sym::Const(b);
    }
  }

  Sym resolve() const {
    if (m_symGraph != nullptr) {
      return m_symGraph->resolve(m_self);
    }
    return m_self;
  }

  Sym resolve(SymGraph *symGraph) const {
    if (symGraph != nullptr) {
      return symGraph->resolve(m_self);
    }
    return m_self;
  }

  bool isSymbolic() const {
    if (m_self.isConstant()) {
      return false;
    }
    assert(!m_self.isSymbolic() || m_symGraph != nullptr);
    return m_symGraph->resolve(m_self).isSymbolic();
  }

  bool isConstant() const {
    if (m_self.isConstant()) {
      return true;
    }
    assert(!m_self.isSymbolic() || m_symGraph != nullptr);
    return m_symGraph->resolve(m_self).isConstant();
  }

  Sym::value_type constant() const {
    if (m_self.isConstant()) {
      return m_self.constant();
    }
    assert(!m_self.isSymbolic() || m_symGraph != nullptr);
    return m_symGraph->resolve(m_self).constant();
  }

  // NOTE: This can definitely be nullptr!
  // Symbolics, which hold a constant, will do
  // not stricly need to hold a valid symGraph pointer.
  SymGraph *graph() const { return m_symGraph; }

  Symbolic() : m_symGraph(nullptr), m_self(Sym::Const(0)) {}

private:
  static Symbolic constBinaryOp(const Symbolic &lhs, const Symbolic &rhs,
                                Sym::value_type v) {
    return Symbolic{selectSymGraph(lhs, rhs), Sym::Const(v)};
  }

  static Symbolic constBinaryOp(const Symbolic &lhs, Sym::value_type v) {
    return Symbolic{lhs.m_symGraph, Sym::Const(v)};
  }

  static SymGraph *selectSymGraph(const Symbolic &lhs, const Symbolic &rhs) {
    SymGraph *symGraph = lhs.m_symGraph;
    if (symGraph == nullptr) {
      symGraph = rhs.m_symGraph;
    }
    return symGraph;
  }

  SymGraph *m_symGraph;
  Sym m_self;
};

} // namespace denox::compiler
