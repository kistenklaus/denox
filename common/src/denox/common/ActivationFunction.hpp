#pragma once
#include "denox/diag/unreachable.hpp"
#include <fmt/core.h>
#include <variant>

namespace denox {

enum class ActivationFunctionKind {
  ReLU,
  LeakyReLU,
  SiLU,
  Swish,
};
struct ActivationFunction_ReLU {};
struct ActivationFunction_LeakyReLU {
  float alpha;
};
struct ActivationFunction_SiLU {};
struct ActivationFunction_Swish {
  float beta; // SiLU beta = 1
};

namespace details {

struct ActivationFunction {
public:
  explicit constexpr ActivationFunction(ActivationFunction_ReLU relu)
      : m_repr{relu} {}
  explicit constexpr ActivationFunction(ActivationFunction_LeakyReLU leakyRelu)
      : m_repr{leakyRelu} {}
  explicit constexpr ActivationFunction(ActivationFunction_SiLU silu)
      : m_repr{silu} {}
  explicit constexpr ActivationFunction(ActivationFunction_Swish swish)
      : m_repr{swish} {}

  ActivationFunctionKind kind() const {
    if (std::holds_alternative<ActivationFunction_ReLU>(m_repr)) {
      return ActivationFunctionKind::ReLU;
    } else if (std::holds_alternative<ActivationFunction_LeakyReLU>(m_repr)) {
      return ActivationFunctionKind::LeakyReLU;
    } else if (std::holds_alternative<ActivationFunction_SiLU>(m_repr)) {
      return ActivationFunctionKind::SiLU;
    } else if (std::holds_alternative<ActivationFunction_Swish>(m_repr)) {
      return ActivationFunctionKind::Swish;
    }
    diag::unreachable("Invalid ActivationFunction enum!");
  }

  const ActivationFunction_ReLU &relu() const {
    assert(kind() == ActivationFunctionKind::ReLU);
    return std::get<ActivationFunction_ReLU>(m_repr);
  }

  const ActivationFunction_LeakyReLU &leaky_relu() const {
    assert(kind() == ActivationFunctionKind::LeakyReLU);
    return std::get<ActivationFunction_LeakyReLU>(m_repr);
  }

  const ActivationFunction_SiLU &silu() const {
    assert(kind() == ActivationFunctionKind::SiLU);
    return std::get<ActivationFunction_SiLU>(m_repr);
  }
  const ActivationFunction_Swish &swish() const {
    assert(kind() == ActivationFunctionKind::Swish);
    return std::get<ActivationFunction_Swish>(m_repr);
  }

private:
  using Repr =
      std::variant<ActivationFunction_ReLU, ActivationFunction_LeakyReLU,
                   ActivationFunction_SiLU, ActivationFunction_Swish>;
  Repr m_repr;
};

}; // namespace details

class ActivationFunction {
public:
  static constexpr details::ActivationFunction ReLU{ActivationFunction_ReLU()};
  static constexpr details::ActivationFunction LeakyReLU(float alpha) {
    return details::ActivationFunction(ActivationFunction_LeakyReLU(alpha));
  }
  static constexpr details::ActivationFunction SiLU{ActivationFunction_SiLU()};
  static constexpr details::ActivationFunction Swish(float beta) {
    return details::ActivationFunction(ActivationFunction_Swish(beta));
  }

  ActivationFunctionKind kind() const { return m_inner.kind(); }

  const ActivationFunction_ReLU &relu() const { return m_inner.relu(); }
  const ActivationFunction_LeakyReLU &leaky_relu() const {
    return m_inner.leaky_relu();
  }
  const ActivationFunction_SiLU &silu() const { return m_inner.silu(); }
  const ActivationFunction_Swish &swish() const { return m_inner.swish(); }

  ActivationFunction(details::ActivationFunction inner) : m_inner(inner) {}

private:
  details::ActivationFunction m_inner;
};

} // namespace denox

template <> struct fmt::formatter<denox::ActivationFunction> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(denox::ActivationFunction f, FormatContext &ctx) const {

    const char *name = nullptr;
    switch (f.kind()) {
    case denox::ActivationFunctionKind::ReLU:
      name = "ReLU";
      break;
    case denox::ActivationFunctionKind::LeakyReLU:
      name = "LeakyReLU";
      break;
    case denox::ActivationFunctionKind::SiLU:
      name = "SiLU";
      break;
    case denox::ActivationFunctionKind::Swish:
      name = "Swish";
      break;
    default:
      name = "?";
      break;
    }
    return fmt::format_to(ctx.out(), "{}", name);
  }
};
