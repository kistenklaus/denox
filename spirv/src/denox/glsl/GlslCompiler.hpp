#pragma once

#include "denox/device_info/DeviceInfo.hpp"
#include "denox/glsl/GlslCompilerInstance.hpp"
#include "denox/glsl/SourceShaCache.hpp"
#include "denox/io/fs/FileCache.hpp"
#include "denox/io/fs/Path.hpp"
#include "denox/spirv/ShaderDebugInfoLevel.hpp"
#include "denox/spirv/SpirvTools.hpp"

namespace denox::spirv {

class GlslCompiler {
public:
  friend class GlslCompilerInstance;

  GlslCompiler(SpirvTools *spvTools, io::FileCache *fileCache,
               const DeviceInfo &deviceInfo,
               SpirvDebugInfoLevel debugInfo = SpirvDebugInfoLevel::Strip,
               bool opt = false);

  GlslCompiler(const GlslCompiler &) = delete;
  GlslCompiler(GlslCompiler &&) = delete;

  GlslCompiler &operator=(const GlslCompiler &) = delete;
  GlslCompiler &operator=(GlslCompiler &&) = delete;
  ~GlslCompiler();

  GlslCompilerInstance read(io::Path sourcePath);

private:
  SpirvTools *m_tools;
  io::FileCache *m_fileCache;

  DeviceInfo m_deviceInfo;
  void * /*TBuiltInResource*/ m_buildInResource;
  SpirvDebugInfoLevel m_debugInfo;
  bool m_optimize;

  SourceShaCache m_shaCache;
};

} // namespace denox::spirv
