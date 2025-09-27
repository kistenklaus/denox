#pragma once

#include "io/fs/Path.hpp"
#include "memory/container/vector.hpp"

namespace denox::compiler {

struct ShaderSourceMeta {
  memory::string shaderName;
  io::Path sourcePath;
};

struct ShaderBinary {
  memory::vector<std::uint32_t> data;
};

struct ShaderReflection {
  // Maybe later we add some extra stuff like reflection information and so on.
  // Might be interessting for generating correct bindings.
};

struct CompiledShader {
  ShaderBinary binary;
  ShaderReflection reflection;
  std::unique_ptr<ShaderSourceMeta> meta;
};

class GlslCompiler;

struct GlslCompilerInstance {
public:
  friend GlslCompiler;
  void enableDenoxPreprocessing();
  void addMacro(memory::string_view name, memory::string_view value);
  void enableNaNClamp();
  CompiledShader compile();

private:
  void *m_self;
};

class GlslCompiler {
public:
  GlslCompiler();
  ~GlslCompiler();
  GlslCompiler(const GlslCompiler &) = delete;
  GlslCompiler &operator=(const GlslCompiler &) = delete;
  GlslCompiler(GlslCompiler &&) = delete;
  GlslCompiler &operator=(GlslCompiler &&) = delete;

  GlslCompilerInstance invoke(io::Path source_path);

private:
  void *m_self;
};

} // namespace denox::compiler
