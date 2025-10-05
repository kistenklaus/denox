#pragma once

#include "device_info/DeviceInfo.hpp"
#include "io/fs/Path.hpp"
#include "memory/container/optional.hpp"
#include "memory/dtype/dtype.hpp"
#include "memory/tensor/ActivationLayout.hpp"
#include "shaders/compiler/ShaderDebugInfoLevel.hpp"
#include "symbolic/Sym.hpp"
#include <type_traits>
#include <variant>
#include <vulkan/vulkan_core.h>

namespace denox::compiler {

enum class SrcType {
  Onnx,
};

struct FusionRules {
  bool enableSliceSliceFusion = false;
  bool enableImplicitConcat = false;
  bool enableConvReluFusion = false;
};

struct TensorShapeExtent {
  memory::optional<std::string> name;
  memory::optional<unsigned int> value;
};

struct TensorShapeDesc {
  TensorShapeExtent channels;
  TensorShapeExtent height;
  TensorShapeExtent width;
};

enum class FeatureState {
  Disable,
  Enable,
  Require
};

inline auto operator<(const FeatureState& lhs, const FeatureState& rhs) {
  using underlying = std::underlying_type_t<FeatureState>;
  return static_cast<underlying>(lhs) < static_cast<underlying>(rhs);
}
inline auto operator<=(const FeatureState& lhs, const FeatureState& rhs) {
  using underlying = std::underlying_type_t<FeatureState>;
  return static_cast<underlying>(lhs) <= static_cast<underlying>(rhs);
}
inline auto operator>(const FeatureState& lhs, const FeatureState& rhs) {
  using underlying = std::underlying_type_t<FeatureState>;
  return static_cast<underlying>(lhs) > static_cast<underlying>(rhs);
}
inline auto operator>=(const FeatureState& lhs, const FeatureState& rhs) {
  using underlying = std::underlying_type_t<FeatureState>;
  return static_cast<underlying>(lhs) >= static_cast<underlying>(rhs);
}

struct Features {
  FeatureState coopmat;
};

struct Options {
  unsigned int dnxVersion;
  SrcType srcType;

  Features features;

  memory::ActivationLayout inputLayout;
  memory::optional<memory::Dtype> inputType;
  TensorShapeDesc inputShape;

  memory::ActivationLayout outputLayout;
  memory::optional<memory::Dtype> outputType;
  TensorShapeDesc outputShape;

  DeviceInfo deviceInfo;
  FusionRules fusionRules;

  ShaderDebugInfoLevel shaderDebugInfo;
  bool optimizeSpirv;

  io::Path cwd;
  memory::optional<io::Path> srcPath;

  bool verbose;
  bool quite;
};

} // namespace denox::compiler
