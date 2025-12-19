#include "frontend/onnx/details/values/Dtype.hpp"

#include "denox/diag/unreachable.hpp"
#include "denox/symbolic/Sym.hpp"
#include <onnx.pb.h>

namespace denox::onnx::details {

std::size_t dtype::details::Dtype::size() const {
  switch (m_kind) {
  case DtypeKind::Int8:
  case DtypeKind::Uint8:
  case DtypeKind::Bool:
    return 1;
  case DtypeKind::Int16:
  case DtypeKind::Uint16:
  case DtypeKind::Float16:
    return 2;
  case DtypeKind::Int32:
  case DtypeKind::Uint32:
  case DtypeKind::Float32:
    return 4;
  case DtypeKind::Int64:
  case DtypeKind::Uint64:
  case DtypeKind::Float64:
    return 8;
  case DtypeKind::Undefined:
    throw std::logic_error("Trying to call Dtype::size with Dtype::Undefined");
  case DtypeKind::String:
    throw std::logic_error("Trying to call Dtype::size with Dtype::String");
  case DtypeKind::Sym:
    return sizeof(Sym);
  }
  denox::compiler::diag::unreachable();
}

memory::optional<denox::memory::Dtype>
dtype::details::Dtype::toDenoxType() const {
  switch (m_kind) {
  case DtypeKind::Float64:
    return memory::Dtype::F64;
  case DtypeKind::Float32:
    return memory::Dtype::F32;
  case DtypeKind::Float16:
    return memory::Dtype::F16;
  // NOTE: All other types are not supported by denox!
  case DtypeKind::Undefined:
  case DtypeKind::Int8:
  case DtypeKind::Int16:
  case DtypeKind::Int32:
  case DtypeKind::Int64:
  case DtypeKind::Uint8:
  case DtypeKind::Uint16:
  case DtypeKind::Uint32:
  case DtypeKind::Uint64:
  case DtypeKind::String:
  case DtypeKind::Bool:
  case DtypeKind::Sym:
    break;
  }
  return memory::nullopt;
}

memory::string_view dtype::details::Dtype::to_string() const {
  switch (m_kind) {
  case DtypeKind::Undefined:
    return "undefined";
  case DtypeKind::Int8:
    return "int8";
  case DtypeKind::Int16:
    return "int16";
  case DtypeKind::Int32:
    return "int32";
  case DtypeKind::Int64:
    return "int64";
  case DtypeKind::Uint8:
    return "uint8";
  case DtypeKind::Uint16:
    return "uint16";
  case DtypeKind::Uint32:
    return "uint32";
  case DtypeKind::Uint64:
    return "uint64";
  case DtypeKind::Float64:
    return "float64";
  case DtypeKind::Float32:
    return "float32";
  case DtypeKind::Float16:
    return "float16";
  case DtypeKind::String:
    return "string";
  case DtypeKind::Bool:
    return "bool";
  case DtypeKind::Sym:
    return "symbolic";
  }
  denox::compiler::diag::unreachable();
}

memory::optional<Dtype> Dtype::parse(std::int32_t dataType) {
  switch (dataType) {
  case ::onnx::TensorProto_DataType_UNDEFINED:
    return Dtype{Undefined};
  case ::onnx::TensorProto_DataType_FLOAT:
    return Dtype{Float32};
  case ::onnx::TensorProto_DataType_FLOAT16:
    return Dtype{Float16};
  case ::onnx::TensorProto_DataType_DOUBLE:
    return Dtype{Float64};
  case ::onnx::TensorProto_DataType_INT8:
    return Dtype{Int8};
  case ::onnx::TensorProto_DataType_INT16:
    return Dtype{Int16};
  case ::onnx::TensorProto_DataType_INT32:
    return Dtype{Int32};
  case ::onnx::TensorProto_DataType_INT64:
    return Dtype{Int64};
  case ::onnx::TensorProto_DataType_UINT8:
    return Dtype{Uint8};
  case ::onnx::TensorProto_DataType_UINT16:
    return Dtype{Uint16};
  case ::onnx::TensorProto_DataType_UINT32:
    return Dtype{Uint32};
  case ::onnx::TensorProto_DataType_UINT64:
    return Dtype{Uint64};
  case ::onnx::TensorProto_DataType_STRING:
    return Dtype{String};
  case ::onnx::TensorProto_DataType_BOOL:
    return Dtype{Bool};
  case ::onnx::TensorProto_DataType_BFLOAT16:
  case ::onnx::TensorProto_DataType_COMPLEX64:
  case ::onnx::TensorProto_DataType_COMPLEX128:
  case ::onnx::TensorProto_DataType_FLOAT8E4M3FN:
  case ::onnx::TensorProto_DataType_FLOAT8E4M3FNUZ:
  case ::onnx::TensorProto_DataType_FLOAT8E5M2:
  case ::onnx::TensorProto_DataType_FLOAT8E5M2FNUZ:
  default:
    return memory::nullopt;
  }
}

memory::string_view Dtype::parse_to_string(std::int32_t dataType) {
  switch (dataType) {
  case ::onnx::TensorProto_DataType_UNDEFINED:
    return "undefined";
  case ::onnx::TensorProto_DataType_FLOAT:
    return "float32";
  case ::onnx::TensorProto_DataType_FLOAT16:
    return "float16";
  case ::onnx::TensorProto_DataType_DOUBLE:
    return "float64";
  case ::onnx::TensorProto_DataType_INT8:
    return "int8";
  case ::onnx::TensorProto_DataType_INT16:
    return "int16";
  case ::onnx::TensorProto_DataType_INT32:
    return "int32";
  case ::onnx::TensorProto_DataType_INT64:
    return "int64";
  case ::onnx::TensorProto_DataType_UINT8:
    return "uint8";
  case ::onnx::TensorProto_DataType_UINT16:
    return "uint16";
  case ::onnx::TensorProto_DataType_UINT32:
    return "uint32";
  case ::onnx::TensorProto_DataType_UINT64:
    return "uint64";
  case ::onnx::TensorProto_DataType_BFLOAT16:
    return "bfloat16";
  case ::onnx::TensorProto_DataType_BOOL:
    return "bool";
  case ::onnx::TensorProto_DataType_STRING:
    return "string";
  case ::onnx::TensorProto_DataType_COMPLEX64:
    return "complex64";
  case ::onnx::TensorProto_DataType_COMPLEX128:
    return "complex128";
  case ::onnx::TensorProto_DataType_FLOAT8E4M3FN:
    return "float8e4m3fn";
  case ::onnx::TensorProto_DataType_FLOAT8E4M3FNUZ:
    return "float8e4m3fnuz";
  case ::onnx::TensorProto_DataType_FLOAT8E5M2:
    return "float8e5m2";
  case ::onnx::TensorProto_DataType_FLOAT8E5M2FNUZ:
    return "float8e5m2fnuz";
  default:
    throw std::runtime_error("Unexpected data_type");
  }
}

memory::optional<denox::memory::Dtype> Dtype::toDenoxType() const {
  return m_type.toDenoxType();
}
bool dtype::details::Dtype::isSignedInt() const {
  switch (m_kind) {
  case DtypeKind::Int8:
  case DtypeKind::Int16:
  case DtypeKind::Int32:
  case DtypeKind::Int64:
    return true;
  case DtypeKind::Undefined:
  case DtypeKind::Uint8:
  case DtypeKind::Uint16:
  case DtypeKind::Uint32:
  case DtypeKind::Uint64:
  case DtypeKind::Float64:
  case DtypeKind::Float32:
  case DtypeKind::Float16:
  case DtypeKind::String:
  case DtypeKind::Bool:
  case DtypeKind::Sym:
    return false;
  }
  denox::compiler::diag::unreachable();
}
bool dtype::details::Dtype::isUnsignedInt() const {
  switch (m_kind) {
  case DtypeKind::Undefined:
  case DtypeKind::Int8:
  case DtypeKind::Int16:
  case DtypeKind::Int32:
  case DtypeKind::Int64:
  case DtypeKind::Float64:
  case DtypeKind::Float32:
  case DtypeKind::Float16:
  case DtypeKind::String:
  case DtypeKind::Bool:
  case DtypeKind::Sym:
    return false;
  case DtypeKind::Uint8:
  case DtypeKind::Uint16:
  case DtypeKind::Uint32:
  case DtypeKind::Uint64:
    return true;
  }
  denox::compiler::diag::unreachable();
}
bool dtype::details::Dtype::isFloat() const {
  switch (m_kind) {
  case DtypeKind::Undefined:
  case DtypeKind::Int8:
  case DtypeKind::Int16:
  case DtypeKind::Int32:
  case DtypeKind::Int64:
  case DtypeKind::Uint8:
  case DtypeKind::Uint16:
  case DtypeKind::Uint32:
  case DtypeKind::Uint64:
  case DtypeKind::String:
  case DtypeKind::Bool:
  case DtypeKind::Sym:
    return false;
  case DtypeKind::Float64:
  case DtypeKind::Float32:
  case DtypeKind::Float16:
    return true;
  }
  denox::compiler::diag::unreachable();
}
bool dtype::details::Dtype::isInteger() const {
  switch (m_kind) {
  case DtypeKind::Int8:
  case DtypeKind::Int16:
  case DtypeKind::Int32:
  case DtypeKind::Int64:
  case DtypeKind::Uint8:
  case DtypeKind::Uint16:
  case DtypeKind::Uint32:
  case DtypeKind::Uint64:
    return true;
  case DtypeKind::Undefined:
  case DtypeKind::Float64:
  case DtypeKind::Float32:
  case DtypeKind::Float16:
  case DtypeKind::String:
  case DtypeKind::Bool:
  case DtypeKind::Sym:
    return false;
  }
  denox::compiler::diag::unreachable();
}
} // namespace denox::onnx::details
