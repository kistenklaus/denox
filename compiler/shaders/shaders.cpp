#include "shaders/shaders.hpp"
#include "shaders/activation/BasicActivationShader.hpp"
#include "shaders/conv/DirectConvShaderCM.hpp"
#include "shaders/copy/CopyTransformShader.hpp"
#include "shaders/pad/MemoryPadShader.hpp"
#include "shaders/pool/BasicPoolShader.hpp"
#include "shaders/slice/MemorySliceShader.hpp"
#include "shaders/upsample/BasicUpsampleShader.hpp"
#include <memory>

namespace denox::compiler::shaders {

std::vector<std::unique_ptr<IShader>> get_all_shaders(GlslCompiler *compiler,
                                                      const Options &options) {
  std::vector<std::unique_ptr<IShader>> shaders;
  shaders.push_back(std::make_unique<compiler::shaders::DirectConvShaderCM>(
      compiler, options));
  shaders.push_back(std::make_unique<compiler::shaders::BasicPoolShader>(
      compiler));
  shaders.push_back(std::make_unique<compiler::shaders::BasicUpsampleShader>(
      compiler, options));
  shaders.push_back(std::make_unique<compiler::shaders::MemoryPadShader>(
      compiler));
  shaders.push_back(std::make_unique<compiler::shaders::MemorySliceShader>(
      compiler));
  shaders.push_back(std::make_unique<compiler::shaders::BasicActivationShader>(
      compiler, options));
  shaders.push_back(std::make_unique<compiler::shaders::CopyTransformShader>(
      compiler, options));
  return shaders;
}
} // namespace denox::compiler::shaders
