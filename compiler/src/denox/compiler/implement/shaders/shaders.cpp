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

  auto direct_conv_cm = std::make_unique<compiler::shaders::DirectConvShaderCM>(
      compiler, options);
  bool cm_supported = direct_conv_cm->supported();
  fmt::println("CM-supported : {}", cm_supported);
  shaders.push_back(std::move(direct_conv_cm));

  shaders.push_back(std::make_unique<compiler::shaders::ConcatConvCMShader>(
      compiler, options));

  if (!cm_supported) {
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
      std::make_unique<compiler::shaders::MemoryPadShader>(compiler, options));

  shaders.push_back(std::make_unique<compiler::shaders::MemorySliceShader>(
      compiler, options));

  shaders.push_back(std::make_unique<compiler::shaders::BasicActivationShader>(
      compiler, options));

  return shaders;
}

} // namespace denox::compiler::shaders
