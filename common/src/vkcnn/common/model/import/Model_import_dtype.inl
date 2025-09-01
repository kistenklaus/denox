#pragma once

#include "vkcnn/common/symbolic/Sym.hpp"
#include "vkcnn/common/tensor/FloatType.hpp"
#include <onnx.pb.h>

#include <optional>
#include <stdexcept>
namespace vkcnn::details {

enum class Dtype {
  Undefined,
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
  // everything else is not supported (currently)
};

static std::optional<Dtype> parse_data_type(std::int32_t dataType) {
  switch (dataType) {
  case onnx::TensorProto_DataType_UNDEFINED:
    return Dtype::Undefined;
  case onnx::TensorProto_DataType_FLOAT:
    return Dtype::Float32;
  case onnx::TensorProto_DataType_FLOAT16:
    return Dtype::Float16;
  case onnx::TensorProto_DataType_DOUBLE:
    return Dtype::Float64;
  case onnx::TensorProto_DataType_INT8:
    return Dtype::Int8;
  case onnx::TensorProto_DataType_INT16:
    return Dtype::Int16;
  case onnx::TensorProto_DataType_INT32:
    return Dtype::Int32;
  case onnx::TensorProto_DataType_INT64:
    return Dtype::Int64;
  case onnx::TensorProto_DataType_UINT8:
    return Dtype::Uint8;
  case onnx::TensorProto_DataType_UINT16:
    return Dtype::Uint16;
  case onnx::TensorProto_DataType_UINT32:
    return Dtype::Uint32;
  case onnx::TensorProto_DataType_UINT64:
    return Dtype::Uint64;
  case onnx::TensorProto_DataType_STRING:
    return Dtype::String;
  case onnx::TensorProto_DataType_BOOL:
    return Dtype::Bool;
  case onnx::TensorProto_DataType_BFLOAT16:
  case onnx::TensorProto_DataType_COMPLEX64:
  case onnx::TensorProto_DataType_COMPLEX128:
  case onnx::TensorProto_DataType_FLOAT8E4M3FN:
  case onnx::TensorProto_DataType_FLOAT8E4M3FNUZ:
  case onnx::TensorProto_DataType_FLOAT8E5M2:
  case onnx::TensorProto_DataType_FLOAT8E5M2FNUZ:
  default:
    return std::nullopt;
  }
}

static std::string_view data_type_to_string(std::int32_t dataType) {
  switch (dataType) {
  case onnx::TensorProto_DataType_UNDEFINED:
    return "undefined";
  case onnx::TensorProto_DataType_FLOAT:
    return "float32";
  case onnx::TensorProto_DataType_FLOAT16:
    return "float16";
  case onnx::TensorProto_DataType_DOUBLE:
    return "float64";
  case onnx::TensorProto_DataType_INT8:
    return "int8";
  case onnx::TensorProto_DataType_INT16:
    return "int16";
  case onnx::TensorProto_DataType_INT32:
    return "int32";
  case onnx::TensorProto_DataType_INT64:
    return "int64";
  case onnx::TensorProto_DataType_UINT8:
    return "uint8";
  case onnx::TensorProto_DataType_UINT16:
    return "uint16";
  case onnx::TensorProto_DataType_UINT32:
    return "uint32";
  case onnx::TensorProto_DataType_UINT64:
    return "uint64";
  case onnx::TensorProto_DataType_BFLOAT16:
    return "bfloat16";
  case onnx::TensorProto_DataType_BOOL:
    return "bool";
  case onnx::TensorProto_DataType_STRING:
    return "string";
  case onnx::TensorProto_DataType_COMPLEX64:
    return "complex64";
  case onnx::TensorProto_DataType_COMPLEX128:
    return "complex128";
  case onnx::TensorProto_DataType_FLOAT8E4M3FN:
    return "float8e4m3fn";
  case onnx::TensorProto_DataType_FLOAT8E4M3FNUZ:
    return "float8e4m3fnuz";
  case onnx::TensorProto_DataType_FLOAT8E5M2:
    return "float8e5m2";
  case onnx::TensorProto_DataType_FLOAT8E5M2FNUZ:
    return "float8e5m2fnuz";
  default:
    throw std::runtime_error("Unexpected data_type");
  }
}

static std::size_t dtype_size(Dtype dtype) {
  switch (dtype) {
  case Dtype::Int8:
  case Dtype::Uint8:
  case Dtype::Bool:
    return 1;
  case Dtype::Int16:
  case Dtype::Uint16:
  case Dtype::Float16:
    return 2;
  case Dtype::Int32:
  case Dtype::Uint32:
  case Dtype::Float32:
    return 4;
  case Dtype::Int64:
  case Dtype::Uint64:
  case Dtype::Float64:
    return 8;
  case Dtype::Undefined:
    throw std::logic_error("Trying to call dtype_size with Dtype::Undefined");
  case Dtype::String:
    throw std::logic_error("Trying to call dtype_size with Dtype::String");
  case Dtype::Sym:
    return sizeof(Sym);
  }
  throw std::runtime_error("Unexpected dtype");
}

static std::string dtype_to_string(Dtype dtype) {
  switch (dtype) {
  case Dtype::Undefined:
    return "undefined";
  case Dtype::Int8:
    return "int8";
  case Dtype::Int16:
    return "int16";
  case Dtype::Int32:
    return "int32";
  case Dtype::Int64:
    return "int64";
  case Dtype::Uint8:
    return "uint8";
  case Dtype::Uint16:
    return "uint16";
  case Dtype::Uint32:
    return "uint32";
  case Dtype::Uint64:
    return "uint64";
  case Dtype::Float64:
    return "float64";
  case Dtype::Float32:
    return "float32";
  case Dtype::Float16:
    return "float16";
  case Dtype::String:
    return "string";
  case Dtype::Bool:
    return "bool";
  case Dtype::Sym:
    return "symbolic";
    break;
  }
  throw std::logic_error("Unexpected dtype");
}

static std::optional<vkcnn::FloatType> dtype_to_float_type(Dtype d) {
  switch (d) {
  case Dtype::Float16:
    return vkcnn::FloatType::F16;
  case Dtype::Float32:
    return vkcnn::FloatType::F32;
  case Dtype::Float64:
    return vkcnn::FloatType::F64;
  case Dtype::Undefined:
    return std::nullopt;
  default:
    break;
  }
  return std::nullopt; // others unsupported as activation dtype
}

} // namespace vkcnn::details
