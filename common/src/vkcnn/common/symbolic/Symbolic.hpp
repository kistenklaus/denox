#pragma once

#include "vkcnn/common/symbolic/SymGraph.hpp"
#include <memory>
namespace vkcnn {

class Symbolic {
public:
  explicit Symbolic(std::shared_ptr<SymGraph> graph, Sym self)
      : m_symGraph(graph), m_self(self) {}

  friend Symbolic operator+(const Symbolic &a, const Symbolic &b) {
    assert(a.m_symGraph.get() == b.m_symGraph.get());
    return Symbolic(a.m_symGraph, a.m_symGraph->add(a.m_self, b.m_self));
  }

  template <typename S>
    requires(std::same_as<S, Sym> || std::is_integral_v<S>)
  friend Symbolic operator+(const Symbolic &a, const S &b) {
    return Symbolic(a.m_symGraph, a.m_symGraph->add(a.m_self, b));
  }

  template <typename S>
  friend Symbolic operator+(const S &a, const Symbolic &b) {
    return Symbolic(b.m_symGraph, b.m_symGraph->add(a, b.m_self));
  }

  friend Symbolic operator-(const Symbolic &a, const Symbolic &b) {
    assert(a.m_symGraph.get() == b.m_symGraph.get());
    return Symbolic(a.m_symGraph, a.m_symGraph->sub(a.m_self, b.m_self));
  }

  template <typename S>
    requires(std::same_as<S, Sym> || std::is_integral_v<S>)
  friend Symbolic operator-(const Symbolic &a, const S &b) {
    return Symbolic(a.m_symGraph, a.m_symGraph->sub(a.m_self, b));
  }

  template <typename S>
  friend Symbolic operator-(const S &a, const Symbolic &b) {
    return Symbolic(b.m_symGraph, b.m_symGraph->sub(a, b.m_self));
  }

  friend Symbolic operator*(const Symbolic &a, const Symbolic &b) {
    assert(a.m_symGraph.get() == b.m_symGraph.get());
    return Symbolic(a.m_symGraph, a.m_symGraph->mul(a.m_self, b.m_self));
  }

  template <typename S>
    requires(std::same_as<S, Sym> || std::is_integral_v<S>)
  friend Symbolic operator*(const Symbolic &a, const S &b) {
    return Symbolic(a.m_symGraph, a.m_symGraph->mul(a.m_self, b));
  }

  template <typename S>
  friend Symbolic operator*(const S &a, const Symbolic &b) {
    return Symbolic(b.m_symGraph, b.m_symGraph->mul(a, b.m_self));
  }

  friend Symbolic operator/(const Symbolic &a, const Symbolic &b) {
    assert(a.m_symGraph.get() == b.m_symGraph.get());
    return Symbolic(a.m_symGraph, a.m_symGraph->div(a.m_self, b.m_self));
  }

  template <typename S>
    requires(std::same_as<S, Sym> || std::is_integral_v<S>)
  friend Symbolic operator/(const Symbolic &a, const S &b) {
    return Symbolic(a.m_symGraph, a.m_symGraph->div(a.m_self, b));
  }

  template <typename S>
  friend Symbolic operator/(const S &a, const Symbolic &b) {
    return Symbolic(b.m_symGraph, b.m_symGraph->div(a, b.m_self));
  }

  friend Symbolic operator%(const Symbolic &a, const Symbolic &b) {
    assert(a.m_symGraph.get() == b.m_symGraph.get());
    return Symbolic(a.m_symGraph, a.m_symGraph->mod(a.m_self, b.m_self));
  }

  template <typename S>
    requires(std::same_as<S, Sym> || std::is_integral_v<S>)
  friend Symbolic operator%(const Symbolic &a, const S &b) {
    return Symbolic(a.m_symGraph, a.m_symGraph->mod(a.m_self, b));
  }

  template <typename S>
  friend Symbolic operator%(const S &a, const Symbolic &b) {
    return Symbolic(b.m_symGraph, b.m_symGraph->mod(a, b.m_self));
  }

  const Sym &operator*() const { return m_self; }

  operator Sym() const {
    return m_self;
  }

private:
  std::shared_ptr<SymGraph> m_symGraph;
  Sym m_self;
};

} // namespace vkcnn
