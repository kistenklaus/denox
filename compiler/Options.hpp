#pragma once

#include "device_info/DeviceInfo.hpp"
#include "io/fs/Path.hpp"
#include "memory/container/optional.hpp"
#include "memory/dtype/dtype.hpp"
#include "memory/tensor/ActivationLayout.hpp"
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

struct Options {
  unsigned int dnxVersion;
  SrcType srcType;
  memory::ActivationLayout inputLayout;
  memory::Dtype inputType;
  memory::ActivationLayout outputLayout;
  memory::Dtype outputType;
  DeviceInfo deviceInfo;
  FusionRules fusionRules;
  io::Path cwd;
  memory::optional<io::Path> srcPath;
};

} // namespace denox::compiler
