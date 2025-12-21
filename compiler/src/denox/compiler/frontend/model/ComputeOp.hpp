#pragma once

#include "denox/compiler/frontend/model/ops/ComputeOpActivation.hpp"
#include "denox/compiler/frontend/model/ops/ComputeOpConcat.hpp"
#include "denox/compiler/frontend/model/ops/ComputeOpConv.hpp"
#include "denox/compiler/frontend/model/ops/ComputeOpPad.hpp"
#include "denox/compiler/frontend/model/ops/ComputeOpPool.hpp"
#include "denox/compiler/frontend/model/ops/ComputeOpSlice.hpp"
#include "denox/compiler/frontend/model/ops/ComputeOpUpsample.hpp"
#include <fmt/core.h>
#include <variant>

namespace denox::compiler {

enum class ComputeOpKind {
  None,
  Conv,
  Activation,
  Upsample,
  Pool,
  Concat,
  Pad,
  Slice,
};

class Model;

class ComputeOp {
public:
  ComputeOp() = default;

  ComputeOp(ComputeOpConv conv) : m_var(std::move(conv)) {}

  ComputeOp(ComputeOpActivation activation) : m_var(std::move(activation)) {}

  ComputeOp(ComputeOpUpsample upsample) : m_var(std::move(upsample)) {}

  ComputeOp(ComputeOpPool pool) : m_var(std::move(pool)) {}

  ComputeOp(ComputeOpConcat concat) : m_var(std::move(concat)) {}

  ComputeOp(ComputeOpPad pad) : m_var(std::move(pad)) {}

  ComputeOp(ComputeOpSlice slice) : m_var(std::move(slice)) {}

  ComputeOpKind tag() const {
    switch (m_var.index()) {
    case 0:
      return ComputeOpKind::None;
    case 1:
      return ComputeOpKind::Conv;
    case 2:
      return ComputeOpKind::Activation;
    case 3:
      return ComputeOpKind::Upsample;
    case 4:
      return ComputeOpKind::Pool;
    case 5:
      return ComputeOpKind::Concat;
    case 6:
      return ComputeOpKind::Pad;
    case 7:
      return ComputeOpKind::Slice;
    default:
      std::abort();
    }
  }

  const ComputeOpConv &conv() const {
    assert(std::holds_alternative<ComputeOpConv>(m_var));
    return std::get<ComputeOpConv>(m_var);
  }

  ComputeOpConv &conv() {
    assert(std::holds_alternative<ComputeOpConv>(m_var));
    return std::get<ComputeOpConv>(m_var);
  }

  const ComputeOpActivation &activation() const {
    assert(std::holds_alternative<ComputeOpActivation>(m_var));
    return std::get<ComputeOpActivation>(m_var);
  }

  ComputeOpActivation &activation() {
    assert(std::holds_alternative<ComputeOpActivation>(m_var));
    return std::get<ComputeOpActivation>(m_var);
  }

  const ComputeOpUpsample &upsample() const {
    assert(std::holds_alternative<ComputeOpUpsample>(m_var));
    return std::get<ComputeOpUpsample>(m_var);
  }

  ComputeOpUpsample &upsample() {
    assert(std::holds_alternative<ComputeOpUpsample>(m_var));
    return std::get<ComputeOpUpsample>(m_var);
  }

  const ComputeOpPool &pool() const {
    assert(std::holds_alternative<ComputeOpPool>(m_var));
    return std::get<ComputeOpPool>(m_var);
  }

  ComputeOpPool &pool() {
    assert(std::holds_alternative<ComputeOpPool>(m_var));
    return std::get<ComputeOpPool>(m_var);
  }

  const ComputeOpConcat &concat() const {
    assert(std::holds_alternative<ComputeOpConcat>(m_var));
    return std::get<ComputeOpConcat>(m_var);
  }

  ComputeOpConcat &concat() {
    assert(std::holds_alternative<ComputeOpConcat>(m_var));
    return std::get<ComputeOpConcat>(m_var);
  }

  const ComputeOpPad &pad() const {
    assert(std::holds_alternative<ComputeOpPad>(m_var));
    return std::get<ComputeOpPad>(m_var);
  }

  ComputeOpPad &pad() {
    assert(std::holds_alternative<ComputeOpPad>(m_var));
    return std::get<ComputeOpPad>(m_var);
  }

  const ComputeOpSlice &slice() const {
    assert(std::holds_alternative<ComputeOpSlice>(m_var));
    return std::get<ComputeOpSlice>(m_var);
  }

  ComputeOpSlice &slice() {
    assert(std::holds_alternative<ComputeOpSlice>(m_var));
    return std::get<ComputeOpSlice>(m_var);
  }

private:
  using Variant =
      std::variant<std::monostate, ComputeOpConv, ComputeOpActivation,
                   ComputeOpUpsample, ComputeOpPool, ComputeOpConcat,
                   ComputeOpPad, ComputeOpSlice>;

  Variant m_var;
};

} // namespace denox::compiler

template <> struct fmt::formatter<denox::compiler::ComputeOp> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::compiler::ComputeOp &op, FormatContext &ctx) const {
    using denox::compiler::ComputeOpKind;

    switch (op.tag()) {
    case ComputeOpKind::None:
      return fmt::format_to(ctx.out(), "None{{}}");

    case ComputeOpKind::Conv:
      return fmt::format_to(ctx.out(), "Conv{}", op.conv());

    case ComputeOpKind::Activation:
      return fmt::format_to(ctx.out(), "Activation{}", op.activation());

    case ComputeOpKind::Upsample:
      return fmt::format_to(ctx.out(), "Upsample{}", op.upsample());

    case ComputeOpKind::Pool:
      return fmt::format_to(ctx.out(), "Pool{}", op.pool());

    case ComputeOpKind::Concat:
      return fmt::format_to(ctx.out(), "Concat");

    case ComputeOpKind::Pad:
      return fmt::format_to(ctx.out(), "Pad{}", op.pad());

    case ComputeOpKind::Slice:
      return fmt::format_to(ctx.out(), "Slice{}", op.slice());
    }

    std::abort(); // unreachable
  }
};
