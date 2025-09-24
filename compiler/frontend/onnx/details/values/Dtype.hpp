#pragma once

#include "diag/unreachable.hpp"
#include "memory/container/optional.hpp"
#include "memory/container/string_view.hpp"
#include "memory/dtype/dtype.hpp"

#include <cstdint>
#include <tuple>
namespace denox::onnx::details {

enum class DtypeKind {
  Undefined = 20,
  Int8,
  Int16,
  Int32,
  Int64,
  Uint8,
  Uint16,
  Uint32,
  Uint64,
  Float64,
  Float32,
  Float16,
  String,
  Bool,
  Sym,
};

namespace dtype::details {
struct Dtype {
  constexpr Dtype(DtypeKind kind) : m_kind(kind) {}

  std::size_t size() const;

  memory::optional<denox::memory::Dtype> toDenoxType() const;
  memory::string_view to_string() const;
  DtypeKind kind() const { return m_kind; }

  friend bool operator==(const Dtype &lhs, const Dtype &rhs) {
    return lhs.m_kind == rhs.m_kind;
  }
  friend bool operator!=(const Dtype &lhs, const Dtype &rhs) {
    return lhs.m_kind != rhs.m_kind;
  }

  bool isSignedInt() const;
  bool isUnsignedInt() const;
  bool isInteger() const;
  bool isFloat() const;

private:
  DtypeKind m_kind;
};
} // namespace dtype::details

class Dtype {
public:
  Dtype(DtypeKind kind) : m_type(kind) {}
  Dtype(dtype::details::Dtype type) : m_type(type) {}
  Dtype() : m_type(DtypeKind::Undefined) {}

  static constexpr dtype::details::Dtype Undefined{DtypeKind::Undefined};
  static constexpr dtype::details::Dtype Int8{DtypeKind::Int8};
  static constexpr dtype::details::Dtype Int16{DtypeKind::Int16};
  static constexpr dtype::details::Dtype Int32{DtypeKind::Int32};
  static constexpr dtype::details::Dtype Int64{DtypeKind::Int64};
  static constexpr dtype::details::Dtype Uint8{DtypeKind::Uint8};
  static constexpr dtype::details::Dtype Uint16{DtypeKind::Uint16};
  static constexpr dtype::details::Dtype Uint32{DtypeKind::Uint32};
  static constexpr dtype::details::Dtype Uint64{DtypeKind::Uint64};
  static constexpr dtype::details::Dtype Float16{DtypeKind::Float16};
  static constexpr dtype::details::Dtype Float32{DtypeKind::Float32};
  static constexpr dtype::details::Dtype Float64{DtypeKind::Float64};
  static constexpr dtype::details::Dtype String{DtypeKind::String};
  static constexpr dtype::details::Dtype Bool{DtypeKind::Bool};
  static constexpr dtype::details::Dtype Sym{DtypeKind::Sym};

  static memory::optional<Dtype> parse(std::int32_t dataType);

  static memory::string_view parse_to_string(std::int32_t dataType);

  std::size_t size() const { return m_type.size(); }
  memory::string_view to_string() const { return m_type.to_string(); }
  DtypeKind kind() const { return m_type.kind(); }

  friend bool operator==(const Dtype &lhs, const Dtype &rhs) {
    return lhs.m_type == rhs.m_type;
  }
  friend bool operator!=(const Dtype &lhs, const Dtype &rhs) {
    return lhs.m_type != rhs.m_type;
  }
  bool isSignedInt() const { return m_type.isSignedInt(); }
  bool isUnsignedInt() const { return m_type.isUnsignedInt(); }
  bool isInteger() const { return m_type.isInteger(); }
  bool isFloat() const { return m_type.isFloat(); }

  memory::optional<memory::Dtype> toDenoxType() const;

private:
  dtype::details::Dtype m_type;
};

} // namespace denox::onnx::details
