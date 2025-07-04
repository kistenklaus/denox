#pragma once

#include <cstddef>
namespace vkcnn {

/// Float precision.
enum class FPrec {
  F16,
  F32,
  F64,
};

static inline std::size_t FPrec_Size(FPrec precision) {
  switch (precision) {
  case FPrec::F16:
    return 2;
  case FPrec::F32:
    return 4;
  case FPrec::F64:
    return 8;
  }
  return -1;
}

} // namespace vkcnn
