#pragma once

#include "Options.hpp"
#include "device_info/DeviceInfo.hpp"
#include "denox/io/fs/Path.hpp"
#include "shaders/compiler/ShaderDebugInfoLevel.hpp"
#include "shaders/compiler/GlslCompilerInstance.hpp"
#include <glslang/Include/ResourceLimits.h>
#include <unordered_map>

namespace denox::compiler {


class GlslCompiler {
public:
  friend class GlslCompilerInstance;
  GlslCompiler(const Options &options);

  GlslCompilerInstance read(io::Path sourcePath);

private:
  DeviceInfo m_deviceInfo;
  TBuiltInResource m_buildInResource;
  ShaderDebugInfoLevel m_debugInfo;
  bool m_optimize;

  std::unordered_map<memory::string, ShaderBinary> m_cache;
};

} // namespace denox::compiler
