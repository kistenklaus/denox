#include "denox/compiler/implement/shaders/shaders.hpp"
#include "denox/compiler/implement/shaders/activation/BasicActivationShader.hpp"
#include "denox/compiler/implement/shaders/conv/ConcatConvCMShader.hpp"
#include "denox/compiler/implement/shaders/conv/DirectConvShader.hpp"
#include "denox/compiler/implement/shaders/conv/DirectConvShaderCM.hpp"
#include "denox/compiler/implement/shaders/copy/CopyTransformShader.hpp"
#include "denox/compiler/implement/shaders/noop/NoOp.hpp"
#include "denox/compiler/implement/shaders/pad/MemoryPadShader.hpp"
#include "denox/compiler/implement/shaders/pool/BasicPoolShader.hpp"
#include "denox/compiler/implement/shaders/slice/MemorySliceShader.hpp"
#include "denox/compiler/implement/shaders/upsample/BasicUpsampleShader.hpp"
#include <memory>

namespace denox::compiler::shaders {

std::vector<std::unique_ptr<IShader>>
get_all_shaders(spirv::GlslCompiler *compiler, const CompileOptions &options) {
  std::vector<std::unique_ptr<IShader>> shaders;

  shaders.push_back(std::make_unique<compiler::NoOp>());

  if (options.features.coopmat && options.deviceInfo.coopmat.supported) {
    shaders.push_back(std::make_unique<compiler::shaders::DirectConvShaderCM>(
        compiler, options));
    shaders.push_back(std::make_unique<compiler::shaders::ConcatConvCMShader>(
        compiler, options));
  } else {
    shaders.push_back(std::make_unique<compiler::shaders::DirectConvShader>(
        compiler, options));
  }
  shaders.push_back(std::make_unique<compiler::shaders::CopyTransformShader>(
      compiler, options));

  shaders.push_back(
      std::make_unique<compiler::shaders::BasicPoolShader>(compiler, options));

  shaders.push_back(std::make_unique<compiler::shaders::BasicUpsampleShader>(
      compiler, options));

  shaders.push_back(
      std::make_unique<compiler::shaders::MemoryPadShader>(compiler));

  shaders.push_back(
      std::make_unique<compiler::shaders::MemorySliceShader>(compiler));

  shaders.push_back(std::make_unique<compiler::shaders::BasicActivationShader>(
      compiler, options));

  return shaders;
}

} // namespace denox::compiler::shaders
