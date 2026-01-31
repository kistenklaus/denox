#pragma once

#include "denox/common/Access.hpp"
#include "denox/common/TensorDataType.hpp"
#include "denox/common/TensorFormat.hpp"
#include "denox/common/TensorStorage.hpp"
#include "denox/memory/container/optional.hpp"
#include <cstdint>
namespace denox {

struct DbTensorBinding {
  uint32_t set;
  uint32_t binding;
  Access access;
  uint64_t byteSize;
  uint16_t alignment;
  TensorFormat format;
  TensorStorage storage;

  // debug info for data sciene.
  memory::optional<uint32_t> width;
  memory::optional<uint32_t> height;
  memory::optional<uint32_t> channels;
  memory::optional<TensorDataType> type;
  bool is_param;
};

} // namespace denox
