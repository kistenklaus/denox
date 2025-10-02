#pragma once

#include "diag/invalid_state.hpp"
#include "diag/unreachable.hpp"
#include "memory/container/hashmap.hpp"
#include "memory/dtype/dtype.hpp"
#include "symbolic/SymGraph.hpp"
#include "symbolic/ssym.hpp"
#include <cstdint>
#include <fmt/format.h>
#include <limits>
#include <stdexcept>
#include <symbolic/Sym.hpp>

namespace denox::compiler {

class PushConstant {
public:
  static PushConstant Dynamic(Sym symbol,
                              memory::Dtype dtype = memory::Dtype::U32) {
    return PushConstant(symbol, dtype);
  }

  static PushConstant Dynamic(sym sym,
                              memory::Dtype dtype = memory::Dtype::U32) {
    return PushConstant(sym.asSym(), dtype);
  }

  PushConstant(Sym symbol, memory::Dtype dtype = memory::Dtype::U32)
      : m_type{0} {
    switch (dtype.kind()) {
    case memory::DtypeKind::F16:
    case memory::DtypeKind::F32:
    case memory::DtypeKind::F64:
      throw std::runtime_error(
          "Floating point push constants are not supported");
    case memory::DtypeKind::U32:
      m_type |= PushConstantType_U32;
      break;
    case memory::DtypeKind::I32:
      m_type |= PushConstantType_I32;
      break;
    }
    if (symbol.isSymbolic()) {
      m_type |= PushConstantType_DynamicSymbolBit;
      m_value.symbol = symbol.sym();
    } else {
      if (m_type & PushConstantType_U32) {
        assert(symbol.constant() >= 0);
        m_value.u32 = static_cast<std::uint32_t>(symbol.constant());
      } else if (m_type & PushConstantType_I32) {
        assert(symbol.constant() <= std::numeric_limits<std::int32_t>::max());
        assert(symbol.constant() >= std::numeric_limits<std::int32_t>::min());
        m_value.i32 = static_cast<std::int32_t>(symbol.constant());
      } else {
        compiler::diag::unreachable();
      }
    }
  }
  PushConstant(std::uint32_t u32) : m_type(PushConstantType_U32) {
    m_value.u32 = u32;
  }

  PushConstant(std::int32_t i32) : m_type(PushConstantType_I32) {
    m_value.i32 = i32;
  }

  memory::string to_string(const SymGraph &symGraph,
                           const memory::hash_map<Sym::symbol, memory::string>
                               &symbolNames = {}) const {
    if (m_type & PushConstantType_DynamicSymbolBit) {
      memory::string str =
          symGraph.to_string(Sym::Symbol(m_value.symbol), symbolNames);
      if (str.size() > 80) {
        return fmt::format("Sym: <very-long-symbolic-expression>");
      } else {
        return fmt::format("Sym: {}", str);
      }
    } else {
      if (m_type & PushConstantType_U32) {
        return fmt::format("{}", m_value.u32);
      } else if (m_type & PushConstantType_I32) {
        return fmt::format("{}", m_value.i32);
      } else {
        compiler::diag::unreachable();
      }
    }
  }

  bool isDynamic() const { return m_type & PushConstantType_DynamicSymbolBit; }

  memory::Dtype type() const {
    if (m_type & PushConstantType_U32) {
      return memory::Dtype::U32;
    } else if (m_type & PushConstantType_I32) {
      return memory::Dtype::I32;
    } else {
      diag::invalid_state();
    }
  }

  Sym::symbol dynamic() const { return m_value.symbol; }
  std::uint32_t u32() const { return m_value.u32; }
  std::int32_t i32() const { return m_value.i32; }

private:
  enum PushConstantType : std::uint64_t {
    PushConstantType_U32 = 1 << 0,
    PushConstantType_I32 = 2 << 0,
    PushConstantType_DynamicSymbolBit = 1 << 8,
  };
  std::underlying_type_t<PushConstantType> m_type;
  union Value {
    std::uint32_t u32;
    std::int32_t i32;
    Sym::symbol symbol;
  } m_value;
};

} // namespace denox::compiler
