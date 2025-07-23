

#include "env.hpp"
#include "vkcnn/common/ActivationFunction.hpp"
#include "vkcnn/common/ops/OpConv.hpp"
#include "vkcnn/common/tensor/ActivationDescriptor.hpp"
#include "vkcnn/common/tensor/ActivationHostTensor.hpp"
#include "vkcnn/common/tensor/FilterHostTensor.hpp"
#include "vkcnn/common/tensor/FilterShape.hpp"
#include "vkcnn/common/tensor/FloatType.hpp"
#include "vkcnn/dev/utils/to_string.hpp"
#include "vkcnn/dev/utils/torch.hpp"
#include "vkcnn/runtime/conv/ConvPipeline.hpp"
#include "vkcnn/shaders/conv/Conv3x3mma16x8x8_CHWC8_RCSKC8_HR_P2.hpp"
#include "vkcnn/shaders/conv/Conv3x3mmaVectorized.hpp"
#include "vkcnn/shaders/conv/ConvTemplate.hpp"

#include <ATen/ops/allclose.h>
#include <ATen/ops/conv2d.h>
#include <fmt/base.h>
#include <glm/fwd.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/torch.h>
#include <unistd.h>

struct ConvTestParams {
  vkcnn::OpConv op;
  std::shared_ptr<vkcnn::shaders::ConvTemplate> conv;
  glm::uvec2 inputSize;
};
class OpTest : public ::testing::TestWithParam<ConvTestParams> {
protected:
  void SetUp() override { ctx = env->ctx(); }

  merian::ContextHandle ctx;
};

std::vector<ConvTestParams> generate_test_params() {

  std::vector<glm::uvec2> inputSizes = {
      glm::uvec2(1999, 1111), //
      // glm::uvec2(960, 540),   //
      // glm::uvec2(480, 270),   //
      // glm::uvec2(240, 135),   //
      glm::uvec2(120, 68), //
      glm::uvec2(120, 72), //
      glm::uvec2(47, 16),  //
      glm::uvec2(31, 7),   //
      glm::uvec2(16, 8),   //
      // glm::uvec2(8, 8),       //
      // glm::uvec2(2, 2),       //
      glm::uvec2(1, 1), //
  };

  std::vector<unsigned int> channelCounts = {
      16, //
      32, //
      48, //
      // 64,  //
      // 80,  //
      // 96,  //
      // 112, //
      128, //
           // 160  //
  };
  std::vector<vkcnn::ActivationLayout> layouts = {
      vkcnn::ActivationLayout::CHWC8,
      vkcnn::ActivationLayout::CHWC16,
  };
  std::vector<vkcnn::FloatType> types = {
      vkcnn::FloatType::F16,
  };
  std::vector<std::optional<vkcnn::ActivationFunction>> activations = {
      std::nullopt,
  };
  std::vector<glm::uvec2> kernelSizes = {glm::uvec2(3, 3)};

  std::vector<vkcnn::OpConv> ops;
  for (const auto &c : channelCounts) {
    for (const auto &k : channelCounts) {
      for (const auto &kernelSize : kernelSizes) {
        for (const auto &inputType : types) {
          for (const auto &outputType : types) {
            for (const auto &filterType : types) {
              for (const auto &inputLayout : layouts) {
                for (const auto &outputLayout : layouts) {
                  for (const auto &activation : activations) {
                    ops.push_back(vkcnn::OpConv{
                        .filterShape = {kernelSize.x, kernelSize.y, c, k},
                        .filterType = filterType,
                        .inputLayout = inputLayout,
                        .inputType = inputType,
                        .outputLayout = outputLayout,
                        .outputType = outputType,
                        .activationFunc = activation});
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  std::vector<std::shared_ptr<vkcnn::shaders::ConvTemplate>> shaders = {
      std::make_shared<vkcnn::shaders::Conv3x3mma16x8x8_CHWC8_RCSKC8_HR_P2>(),
      std::make_shared<vkcnn::shaders::Conv3x3mmaVectorized>(
          glm::uvec3(16, 8, 8)),
      std::make_shared<vkcnn::shaders::Conv3x3mmaVectorized>(
          glm::uvec3(16, 16, 16)),
      // std::make_shared<vkcnn::shaders::Conv3x3mma16x8x8_CHWC8_RSCKC8_NR_P2>(),

  };

  std::vector<ConvTestParams> params;
  for (const auto &op : ops) {
    for (const auto &shader : shaders) {
      if (shader->supports(op)) {
        for (const auto &inputSize : inputSizes) {
          params.push_back(ConvTestParams{op, shader, inputSize});
        }
      }
    }
  }
  return params;
}

TEST_P(OpTest, Shader) {
  const auto &param = GetParam();
  if (!param.conv->supports(param.op)) {
    GTEST_FAIL();
  }

  auto sourceOpt = param.conv->specialize(param.op);
  ASSERT_TRUE(sourceOpt.has_value());
  auto &source = sourceOpt.value();

  ::torch::Device device(::torch::cuda::is_available() ? ::torch::kCUDA
                                                       : ::torch::kCPU);

  ::torch::manual_seed(43);
  ::torch::Tensor filterTorch =
      ::torch::rand({param.op.filterShape.k, param.op.filterShape.c,
                     param.op.filterShape.r, param.op.filterShape.s},
                    ::torch::TensorOptions()
                        .dtype(vkcnn::torch::fromType(param.op.filterType))
                        .device(device));
  vkcnn::FilterHostTensor filterHost = vkcnn::torch::toFilter(
      filterTorch, source.filterLayout(), param.op.filterType);

  vkcnn::runtime::FilterDeviceTensor filterDevice(filterHost.desc(),
                                                  env->alloc());

  ::torch::Tensor inputTorch = ::torch::rand(
      {param.op.filterShape.c, param.inputSize.y, param.inputSize.x},
      ::torch::TensorOptions()
          .dtype(vkcnn::torch::fromType(param.op.inputType))
          .device(device));

  vkcnn::ActivationHostTensor inputHost = vkcnn::torch::toActivation(
      inputTorch, param.op.inputLayout, param.op.inputType);

  vkcnn::runtime::ConvPipeline pipe{env->ctx(), env->sc(), source,
                                    filterDevice};

  vkcnn::runtime::ActivationDeviceTensor inputDevice(inputHost.desc(),
                                                     env->alloc());

  vkcnn::runtime::ActivationDeviceTensor outputDevice(
      vkcnn::ActivationDescriptor{.shape = {param.inputSize.x,
                                            param.inputSize.y,
                                            param.op.filterShape.k},
                                  .layout = param.op.outputLayout,
                                  .type = param.op.outputType},
      env->alloc());

  merian::CommandPoolHandle cmdPool =
      std::make_shared<merian::CommandPool>(env->queue());

  auto cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
  cmd->begin();

  inputDevice.upload(cmd, inputHost);
  filterDevice.upload(cmd, filterHost);
  pipe.run(cmd, inputDevice, outputDevice);
  auto download = outputDevice.download(cmd);

  cmd->end();
  env->queue()->submit_wait(cmd);

  vkcnn::ActivationHostTensor outputHost = download.complete();

  ::torch::Tensor outputTorch = vkcnn::torch::fromActivation(outputHost);

  ::torch::Tensor outputTorchRef =
      ::torch::conv2d(inputTorch, filterTorch, std::nullopt, {1, 1}, {1, 1})
          .unsqueeze(0);

  bool allClose = ::torch::allclose(outputTorch, outputTorchRef, 1e-2, 1e-2);
  EXPECT_TRUE(allClose);
}

std::string convTestName(const testing::TestParamInfo<ConvTestParams> &info) {
  const auto &op = info.param.op;
  const auto &inputSize = info.param.inputSize;
  return fmt::format(
      "{}__conv{}x{}{}__{}x{}x{}{}x{}{}___{}_{}", info.param.conv->name(),
      op.filterShape.s, op.filterShape.r, vkcnn::to_string(op.filterType),
      inputSize.x, inputSize.y, op.filterShape.c,
      vkcnn::to_string(op.inputType), op.filterShape.k,
      vkcnn::to_string(op.outputType), vkcnn::to_string(op.inputLayout),
      vkcnn::to_string(op.outputLayout));
}

INSTANTIATE_TEST_SUITE_P(Conv, OpTest,
                         ::testing::ValuesIn(generate_test_params()),
                         convTestName);
