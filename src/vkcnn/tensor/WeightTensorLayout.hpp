#pragma once

namespace vkcnn {

/// K: Output channels
/// C: Input channels
/// S: Kernel width
/// R: Kernel height
enum class WeightTensorLayout {
  KRSC,
  KCRS,
  RSCK,
  RSKC,
  RSCKC8,
  RCSKC8,
};

} // namespace vkcnn
