#pragma once

#include "denox/compiler.h"
#include "denox/diag/unreachable.hpp"
#include "denox/memory/dtype/dtype.hpp"
#include <optional>
namespace denox::api {

inline std::optional<denox::memory::Dtype>
DenoxDataType_to_Dtype(DenoxDataType dtype) {
  switch (dtype) {
  case DENOX_DATA_TYPE_AUTO:
    return std::nullopt;
  case DENOX_DATA_TYPE_F16:
    return denox::memory::Dtype::F16;
  case DENOX_DATA_TYPE_F32:
    return denox::memory::Dtype::F32;
  case DENOX_DATA_TYPE_U8:
    return std::nullopt;
  case DENOX_DATA_TYPE_U16:
    return std::nullopt;
  case DENOX_DATA_TYPE_U32:
    return denox::memory::Dtype::U32;
  case DENOX_DATA_TYPE_F64:
    return denox::memory::Dtype::F64;
  case DENOX_DATA_TYPE_I8:
    return std::nullopt;
  case DENOX_DATA_TYPE_I16:
    return std::nullopt;
  case DENOX_DATA_TYPE_I32:
    return denox::memory::Dtype::I32;
  }
  denox::compiler::diag::unreachable();
}

inline DenoxDataType Dtype_to_DenoxDataType(denox::memory::Dtype dtype) {
  switch (dtype.kind()) {
  case memory::DtypeKind::F16:
    return DENOX_DATA_TYPE_F16;
  case memory::DtypeKind::F32:
    return DENOX_DATA_TYPE_F32;
  case memory::DtypeKind::F64:
    return DENOX_DATA_TYPE_F64;
  case memory::DtypeKind::U32:
    return DENOX_DATA_TYPE_U32;
  case memory::DtypeKind::I32:
    return DENOX_DATA_TYPE_I32;
  }
  denox::compiler::diag::unreachable();
}

} // namespace denox::api
