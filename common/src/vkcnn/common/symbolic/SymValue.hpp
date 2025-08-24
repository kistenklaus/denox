#pragma once

#include <cassert>
#include <cstdint>
namespace vkcnn {

struct SymValue {

  static constexpr SymValue Var() { return SymValue{true, 0}; }

  static constexpr SymValue Const(std::uint64_t v) {
    return SymValue{false, v};
  }

  bool isConstant() const { return !m_var; }

  bool isVar() const { return m_var; }

  std::uint64_t value() const {
    assert(!m_var);
    return m_value;
  }

private:
  SymValue(bool var, std::uint64_t v) : m_var(var), m_value(v) {}
  std::uint64_t m_var : 1; // if true => is a variable or a intermediate!
  std::uint64_t m_value : 63;
};

}; // namespace vkcnn
