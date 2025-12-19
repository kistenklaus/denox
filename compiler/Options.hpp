#pragma once

#include "device_info/DeviceInfo.hpp"
#include "denox/io/fs/Path.hpp"
#include "denox/memory/container/optional.hpp"
#include "denox/memory/dtype/dtype.hpp"
#include "denox/memory/tensor/ActivationLayout.hpp"
#include "shaders/compiler/ShaderDebugInfoLevel.hpp"
#include <type_traits>
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

enum class FeatureState { Disable, Enable, Require };

inline auto operator<(const FeatureState &lhs, const FeatureState &rhs) {
  using underlying = std::underlying_type_t<FeatureState>;
  return static_cast<underlying>(lhs) < static_cast<underlying>(rhs);
}
inline auto operator<=(const FeatureState &lhs, const FeatureState &rhs) {
  using underlying = std::underlying_type_t<FeatureState>;
  return static_cast<underlying>(lhs) <= static_cast<underlying>(rhs);
}
inline auto operator>(const FeatureState &lhs, const FeatureState &rhs) {
  using underlying = std::underlying_type_t<FeatureState>;
  return static_cast<underlying>(lhs) > static_cast<underlying>(rhs);
}
inline auto operator>=(const FeatureState &lhs, const FeatureState &rhs) {
  using underlying = std::underlying_type_t<FeatureState>;
  return static_cast<underlying>(lhs) >= static_cast<underlying>(rhs);
}

struct Features {
  FeatureState coopmat;
};

struct DescriptorPolicy {
  // sets should be selected in order.
  uint32_t set;
};

struct DescriptorPolicies {
  // policy is selected in order, i.e. first check input then output and so on.
  DescriptorPolicy inputPolicy;
  DescriptorPolicy outputPolicy;
  DescriptorPolicy paramPolicy;
  DescriptorPolicy readPolicy;
  DescriptorPolicy writePolicy;
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
  bool skipSpirvCompile;

  DescriptorPolicies descriptorPolicies;

  io::Path cwd;
  memory::optional<io::Path> srcPath;

  bool verbose;
  bool quite;
  bool summarize;
};

} // namespace denox::compiler
