#pragma once

#include "denox/diag/logging.hpp"
#include "denox/spirv/CompilationError.hpp"
#include "denox/spirv/SpirvBinary.hpp"
#include <exception>
#include <variant>

namespace denox::spirv {

class GlslCompilerInstance;

class CompilationResult {
public:
  friend class GlslCompilerInstance;

  operator bool() const noexcept { return isOk(); }
  bool isErr() const noexcept {
    return std::holds_alternative<CompilationError>(m_repr);
  }
  bool isOk() const noexcept {
    return std::holds_alternative<SpirvBinary>(m_repr);
  }

  const SpirvBinary &binary() const { return std::get<SpirvBinary>(m_repr); }

  const CompilationError &error() const {
    return std::get<CompilationError>(m_repr);
  }

  SpirvBinary operator*() const {
    if (isOk()) {
      return std::get<SpirvBinary>(m_repr);
    } else {
      DENOX_ERROR(error().msg);
      std::terminate();
    }
  }

private:
  CompilationResult(SpirvBinary binary) : m_repr(std::move(binary)) {}
  CompilationResult(CompilationError err) : m_repr(std::move(err)) {}

  std::variant<SpirvBinary, CompilationError> m_repr;
};

} // namespace denox::compiler
