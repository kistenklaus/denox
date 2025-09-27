#pragma once

#include "device_info/DeviceInfo.hpp"
#include "io/fs/Path.hpp"
#include "memory/container/optional.hpp"
#include "memory/container/vector.hpp"
#include "memory/dtype/dtype.hpp"
#include "memory/tensor/ActivationLayout.hpp"
#include <vulkan/vulkan_core.h>

namespace denox::compiler {

enum class SrcType {
  Onnx,
};


struct Options {
  unsigned int dnxVersion;
  SrcType srcType;
  memory::ActivationLayout inputLayout;
  memory::Dtype inputType;
  memory::ActivationLayout outputLayout;
  memory::Dtype outputType;
  DeviceInfo deviceInfo;

  io::Path cwd;
  memory::optional<io::Path> srcPath;
};

} // namespace denox::compiler
