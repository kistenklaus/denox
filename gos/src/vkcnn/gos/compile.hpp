#pragma once

#include "vkcnn/common/model/CompileModel.hpp"
#include "vkcnn/common/model/Model.hpp"
#include "vkcnn/common/mr/MemoryRequirements.hpp"
#include <shaderc/shaderc.hpp>
namespace vkcnn {

CompiledModel compile(const Model &graph);

} // namespace vkcnn

static void foo() {

  vkcnn::Model nn;
  auto in = nn.input(3);
  in.setLayout(vkcnn::ActivationLayout::HWC);
  in.setType(vkcnn::FloatType::F16);
  auto x = nn.Conv3x3(in, 32);

  auto y = nn.ReLU(x);
  auto z = nn.MaxPool(y, glm::uvec2(2, 2));

  auto a = nn.NearestUpsample(z, 2);
  auto a2 = nn.concat(a, y);
  auto b = nn.Conv3x3(a2, 3);
  auto c = nn.ReLU(b);

  c.setLayout(vkcnn::ActivationLayout::HWC);
  c.setType(vkcnn::FloatType::F16);
  nn.output(c);

  auto cnn = vkcnn::compile(nn);
}
