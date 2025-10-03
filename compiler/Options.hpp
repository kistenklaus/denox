#pragma once

#include "device_info/DeviceInfo.hpp"
#include "io/fs/Path.hpp"
#include "memory/container/optional.hpp"
#include "memory/dtype/dtype.hpp"
#include "memory/tensor/ActivationLayout.hpp"
#include "shaders/compiler/ShaderDebugInfoLevel.hpp"
#include "symbolic/Sym.hpp"
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

struct TensorShapeDesc {
  std::variant<std::monostate, memory::string, unsigned int> channels;
  std::variant<std::monostate, memory::string, unsigned int> width;
  std::variant<std::monostate, memory::string, unsigned int> height;
};

struct Options {
  unsigned int dnxVersion;
  SrcType srcType;
  memory::ActivationLayout inputLayout;
  memory::Dtype inputType;
  TensorShapeDesc inputShape;
  memory::ActivationLayout outputLayout;
  memory::Dtype outputType;
  TensorShapeDesc outputShape;
  DeviceInfo deviceInfo;
  FusionRules fusionRules;
  ShaderDebugInfoLevel shaderDebugInfo;
  bool optimizeSpirv;
  io::Path cwd;
  memory::optional<io::Path> srcPath;
};

} // namespace denox::compiler
