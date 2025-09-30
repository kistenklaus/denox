#pragma once

#include "diag/logging.hpp"
#include "shaders/compiler/CompilationError.hpp"
#include "shaders/compiler/ShaderBinary.hpp"
#include <exception>
#include <stdexcept>
#include <variant>

namespace denox::compiler {

class GlslCompilerInstance;

class CompilationResult {
public:
  friend class GlslCompilerInstance;

  operator bool() const noexcept { return isOk(); }
  bool isErr() const noexcept {
    return std::holds_alternative<CompilationError>(m_repr);
  }
  bool isOk() const noexcept {
    return std::holds_alternative<ShaderBinary>(m_repr);
  }

  const ShaderBinary &binary() const { return std::get<ShaderBinary>(m_repr); }

  const CompilationError &error() const {
    return std::get<CompilationError>(m_repr);
  }

  ShaderBinary operator*() const {
    if (isOk()) {
      return std::get<ShaderBinary>(m_repr);
    } else {
      DENOX_ERROR(error().msg);
      std::terminate();
    }
  }

private:
  CompilationResult(ShaderBinary binary) : m_repr(std::move(binary)) {}
  CompilationResult(CompilationError err) : m_repr(std::move(err)) {}

  std::variant<ShaderBinary, CompilationError> m_repr;
};

} // namespace denox::compiler
