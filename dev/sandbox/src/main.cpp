
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "merian/vk/shader/shader_compiler_system_glslc.hpp"
#include "merian/vk/utils/profiler.hpp"
#include "vkcnn/common/ActivationFunction.hpp"
#include "vkcnn/common/FilterMode.hpp"
#include "vkcnn/common/PoolFunction.hpp"
#include "vkcnn/common/containers/small_vector.hpp"
#include "vkcnn/common/hypergraph/NodeId.hpp"
#include "vkcnn/common/ops/OpActivation.hpp"
#include "vkcnn/common/ops/OpCopy.hpp"
#include "vkcnn/common/ops/OpPool.hpp"
#include "vkcnn/common/ops/OpUpsample.hpp"
#include "vkcnn/common/shader/ActivationShaderSource.hpp"
#include "vkcnn/common/shader/ConvShaderSource.hpp"
#include "vkcnn/common/symbolic/SymAdjGraph.hpp"
#include "vkcnn/common/symbolic/SymGraph.hpp"
#include "vkcnn/common/symbolic/modsolve.hpp"
#include "vkcnn/common/tensor/ActivationHostTensor.hpp"
#include "vkcnn/common/tensor/ActivationLayout.hpp"
#include "vkcnn/common/tensor/BiasHostTensor.hpp"
#include "vkcnn/common/tensor/FilterHostTensor.hpp"
#include "vkcnn/common/tensor/FloatType.hpp"
#include "vkcnn/dev/utils/merian.hpp"
#include "vkcnn/dev/utils/tensor_algorithms.hpp"
#include "vkcnn/runtime/activation/ActivationPipeline.hpp"
#include "vkcnn/runtime/conv/ConvPipeline.hpp"
#include "vkcnn/runtime/copy/CopyPipeline.hpp"
#include "vkcnn/runtime/pool/PoolPipeline.hpp"
#include "vkcnn/runtime/tensor/ActivationDeviceTensor.hpp"
#include "vkcnn/runtime/tensor/FilterDeviceTensor.hpp"
#include "vkcnn/runtime/upsample/UpsamplePipeline.hpp"
#include "vkcnn/shaders/activation/ActivationShader.hpp"
#include "vkcnn/shaders/conv/DirectConvShader.hpp"
#include "vkcnn/shaders/copy/CopyTransformShader.hpp"
#include "vkcnn/shaders/pool/DirectPoolShader.hpp"
#include "vkcnn/shaders/upsample/DirectUpsampleShader.hpp"
#include <cassert>
#include <fmt/base.h>
#include <print>

void conv_sandbox() {
  // Setup
  merian::ContextHandle context = vkcnn::merian::createContext();

  merian::ShaderCompilerHandle shaderCompiler =
      std::make_shared<merian::SystemGlslcCompiler>(context);
  auto resources = context->get_extension<merian::ExtensionResources>();
  merian::ResourceAllocatorHandle deviceAlloc = resources->resource_allocator();

  merian::QueueHandle queue = context->get_queue_GCT();
  merian::CommandPoolHandle cmdPool =
      std::make_shared<merian::CommandPool>(queue);

  merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context);
  merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
      std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context);
  query_pool->reset();
  profiler->set_query_pool(query_pool);

  const unsigned int W = 1920;
  const unsigned int H = 1080;
  const unsigned int C = 32;
  const unsigned int K = 32;

  const unsigned int R = 3;
  const unsigned int S = 3;

  const glm::uvec2 stride = glm::uvec2(1, 1);
  const glm::uvec2 padding = glm::uvec2(1, 1);
  const bool useBias = false;

  const vkcnn::ActivationLayout inLayout = vkcnn::ActivationLayout::CHWC8;
  const vkcnn::ActivationLayout outLayout = vkcnn::ActivationLayout::CHWC8;

  const vkcnn::FloatType outType = vkcnn::FloatType::F16;
  const vkcnn::FloatType inType = vkcnn::FloatType::F16;
  const vkcnn::FloatType filterType = vkcnn::FloatType::F16;
  const vkcnn::FloatType biasType = vkcnn::FloatType::F16;
  const vkcnn::FloatType arithmeticType = vkcnn::FloatType::F16;

  std::optional<vkcnn::ActivationFunction> activationFunction = std::nullopt;

  const glm::uvec3 cmShape = glm::uvec3(16, 16, 16);
  const glm::uvec3 sgTile = glm::uvec3(2, 2, 2);
  const glm::uvec2 wgTile = glm::uvec2(8, 1);

  vkcnn::ActivationHostTensor outputHost{
      {vkcnn::ActivationShape{W, H, K}, outLayout, outType}};

  vkcnn::shaders::DirectConvShader conv{cmShape, sgTile, wgTile, true};

  vkcnn::ConvShaderSource convSrc =
      conv.specialize(vkcnn::OpConv{{S, R, C, K},
                                    filterType, // filter type
                                    (useBias ? std::optional(biasType)
                                             : std::nullopt), // bias type
                                    inLayout,
                                    inType, // input type
                                    outLayout,
                                    outType,
                                    activationFunction,
                                    arithmeticType,
                                    stride,
                                    padding})
          .value();

  // ::torch::manual_seed(42);
  // ::torch::Tensor filterTorch = ::torch::rand(
  //     {K, C, R, S}, ::torch::TensorOptions()
  //                       .dtype(vkcnn::torch::fromType(convSrc.filterType()))
  //                       .device(::torch::kCUDA));
  //
  vkcnn::FilterHostTensor filterHost{convSrc.filterDesc()};

  for (unsigned int c = 0; c < C; ++c) {
    for (unsigned int k = 0; k < K; ++k) {
      for (unsigned int r = 0; r < R; ++r) {
        for (unsigned int s = 0; s < S; ++s) {
          filterHost.at(s, r, c, k) = 1.0;
        }
      }
    }
  }

  // vkcnn::FilterHostTensor filterHost = vkcnn::torch::toFilter(
  //     filterTorch, convSrc.filterLayout(), convSrc.filterType());

  // ::torch::Tensor inputTorch = ::torch::rand(
  //     {C, H, W}, ::torch::TensorOptions()
  //                    .dtype(vkcnn::torch::fromType(convSrc.inputType()))
  //                    .device(::torch::kCUDA));

  std::optional<vkcnn::BiasHostTensor> biasHost;

  if (useBias) {
    biasHost.emplace(*convSrc.biasDesc());
    for (uint k = 0; k < K; ++k) {
      biasHost.value().at(k) = static_cast<float>(0);
    }
  }

  // vkcnn::ActivationHostTensor inputHost = vkcnn::torch::toActivation(
  //     inputTorch, convSrc.inputLayout(), convSrc.inputType());
  vkcnn::ActivationHostTensor inputHost{
      {{W, H, C}, convSrc.inputLayout(), convSrc.inputType()}};
  for (unsigned int h = 0; h < H; ++h) {
    for (unsigned int w = 0; w < W; ++w) {
      for (unsigned int c = 0; c < C; ++c) {
        inputHost.at(w, h, c) = 1.0;
      }
    }
  }

  // =========== RUNTIME ==============

  vkcnn::runtime::ActivationDeviceTensor inputDevice{inputHost.desc(),
                                                     deviceAlloc};

  vkcnn::runtime::ActivationDeviceTensor outputDevice{outputHost.desc(),
                                                      deviceAlloc};

  vkcnn::runtime::FilterDeviceTensor filterDevice(filterHost.desc(),
                                                  deviceAlloc);

  std::optional<vkcnn::runtime::BiasDeviceTensor> biasDevice;
  if (biasHost.has_value()) {
    biasDevice.emplace(biasHost->desc(), deviceAlloc);
  }

  vkcnn::runtime::ConvPipeline convPipe{context, shaderCompiler, convSrc,
                                        filterDevice, biasDevice};

  auto cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
  cmd->begin();

  if (biasDevice.has_value()) {
    biasDevice->upload(cmd, *biasHost);
  }
  filterDevice.upload(cmd, filterHost);
  inputDevice.upload(cmd, inputHost);
  outputDevice.zero(cmd);

  for (unsigned int i = 0; i < 10; ++i) {
    MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "conv-pipe");
    convPipe.run(cmd, inputDevice, outputDevice);
  }

  auto download = outputDevice.download(cmd);

  cmd->end();
  queue->submit_wait(cmd);

  download.complete(outputHost);

  // ::torch::Tensor outputTorch = vkcnn::torch::fromActivation(outputHost);
  //
  // ::torch::Tensor outputTorchRef =
  //     ::torch::conv2d(inputTorch, filterTorch, std::nullopt, {1, 1}, {1, 1});

  // vkcnn::ActivationHostTensor outRef =
  //     vkcnn::torch::toActivation(outputTorchRef);
  //
  // ::torch::Tensor diffTorch = outputTorchRef - outputTorch;

  // vkcnn::ActivationHostTensor diff = vkcnn::torch::toActivation(diffTorch);

  if (W <= 32) {
    fmt::println("REF:");
    // vkcnn::tensor_algo::printActivation(outRef);
    fmt::println("OUT:");
    vkcnn::tensor_algo::printActivation(outputHost);
    fmt::println("DIFF:");
    // vkcnn::tensor_algo::printActivation(diff);
  }

  profiler->collect(true, false);
  fmt::println("{}", merian::Profiler::get_report_str(profiler->get_report()));
  double lat = profiler->get_report().gpu_total();
  std::size_t mem = 0;
  if (biasHost.has_value()) {
    mem += biasHost->byteSize();
  }
  mem += inputHost.byteSize();
  mem += filterHost.byteSize();
  mem += outputHost.byteSize();

  double throughput = mem / (lat * 1e-3);
  fmt::println("Throughput: {}GB/s", throughput * 1e-9);

  // for (uint b = 0; b < 16; ++b) {
  //   fmt::println("[{}]: 0x{:X}", b, outputHost.span()[b]);
  // }

  // vkcnn::tensor_algo::printActivation(outputHost);
}

void copy_transform_sandbox() {
  merian::ContextHandle context = vkcnn::merian::createContext();

  merian::ShaderCompilerHandle shaderCompiler =
      std::make_shared<merian::SystemGlslcCompiler>(context);
  auto resources = context->get_extension<merian::ExtensionResources>();
  merian::ResourceAllocatorHandle deviceAlloc = resources->resource_allocator();

  merian::QueueHandle queue = context->get_queue_GCT();
  merian::CommandPoolHandle cmdPool =
      std::make_shared<merian::CommandPool>(queue);

  merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context);
  merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
      std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context);
  query_pool->reset();
  profiler->set_query_pool(query_pool);

  const unsigned int W = 1920;
  const unsigned int H = 1080;
  const unsigned int C = 8;
  const unsigned int IN_C = C;
  const unsigned int OUT_C = C;

  unsigned int outChannelOffset = 0;
  unsigned int inChannelOffset = 0;

  const vkcnn::ActivationLayout inLayout = vkcnn::ActivationLayout::CHWC8;
  const vkcnn::ActivationLayout outLayout = vkcnn::ActivationLayout::CHWC8;

  const vkcnn::FloatType outType = vkcnn::FloatType::F16;
  const vkcnn::FloatType inType = vkcnn::FloatType::F16;

  vkcnn::shaders::CopyTransformShader copyTransform{};

  vkcnn::ActivationHostTensor inputHost{
      {vkcnn::ActivationShape{W, H, IN_C}, inLayout, inType}};
  vkcnn::ActivationHostTensor outputHost{
      {vkcnn::ActivationShape{W, H, OUT_C}, outLayout, outType}};

  vkcnn::OpCopy copy{inLayout,        inType,           IN_C,
                     inChannelOffset, outLayout,        outType,
                     OUT_C,           outChannelOffset, C};

  vkcnn::CopyShaderSource copySrc = copyTransform.do_specialize(copy);

  for (unsigned int h = 0; h < H; ++h) {
    for (unsigned int w = 0; w < W; ++w) {
      for (unsigned int c = 0; c < IN_C; ++c) {
        inputHost.at(w, h, c) = static_cast<float>(c + 1);
      }
    }
  }

  // =========== RUNTIME ==============

  vkcnn::runtime::ActivationDeviceTensor inputDevice{inputHost.desc(),
                                                     deviceAlloc};

  vkcnn::runtime::ActivationDeviceTensor outputDevice{outputHost.desc(),
                                                      deviceAlloc};

  vkcnn::runtime::CopyPipeline copyPipe{context, shaderCompiler, copySrc};

  auto cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
  cmd->begin();

  inputDevice.upload(cmd, inputHost);
  outputDevice.zero(cmd);

  for (unsigned int i = 0; i < 100; ++i) {
    MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "copy-transform");
    copyPipe.run(cmd, inputDevice, outputDevice);
  }

  auto download = outputDevice.download(cmd);

  cmd->end();
  queue->submit_wait(cmd);

  download.complete(outputHost);

  // ::torch::Tensor outputTorch = vkcnn::torch::fromActivation(outputHost);
  //
  // ::torch::Tensor outputTorchRef =
  //     ::torch::conv2d(inputTorch, filterTorch, std::nullopt, {1, 1}, {1, 1});

  // vkcnn::ActivationHostTensor outRef =
  //     vkcnn::torch::toActivation(outputTorchRef);
  //
  // ::torch::Tensor diffTorch = outputTorchRef - outputTorch;

  // vkcnn::ActivationHostTensor diff = vkcnn::torch::toActivation(diffTorch);

  if (W <= 32) {
    fmt::println("REF:");
    // vkcnn::tensor_algo::printActivation(outRef);
    fmt::println("OUT:");
    vkcnn::tensor_algo::printActivation(outputHost);
    fmt::println("DIFF:");
    // vkcnn::tensor_algo::printActivation(diff);
  }

  profiler->collect(true, false);
  fmt::println("{}", merian::Profiler::get_report_str(profiler->get_report()));
  double lat = profiler->get_report().gpu_total();
  std::size_t mem = 0;
  mem +=
      inputHost.shape().h * inputHost.shape().w * C * inputHost.type().size();
  mem += outputHost.shape().h * outputHost.shape().w * C *
         outputHost.type().size();

  double throughput = mem / (lat * 1e-3);
  fmt::println("Throughput: {}GB/s", throughput * 1e-9);
}

void activation_sandbox() {
  merian::ContextHandle context = vkcnn::merian::createContext();

  merian::ShaderCompilerHandle shaderCompiler =
      std::make_shared<merian::SystemGlslcCompiler>(context);
  auto resources = context->get_extension<merian::ExtensionResources>();
  merian::ResourceAllocatorHandle deviceAlloc = resources->resource_allocator();

  merian::QueueHandle queue = context->get_queue_GCT();
  merian::CommandPoolHandle cmdPool =
      std::make_shared<merian::CommandPool>(queue);

  merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context);
  merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
      std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context);
  query_pool->reset();
  profiler->set_query_pool(query_pool);

  const unsigned int W = 1920;
  const unsigned int H = 1080;
  const unsigned int C = 32;

  const vkcnn::ActivationLayout inLayout = vkcnn::ActivationLayout::HWC;
  const vkcnn::ActivationLayout outLayout = vkcnn::ActivationLayout::HWC;

  const vkcnn::FloatType outType = vkcnn::FloatType::F16;
  const vkcnn::FloatType inType = vkcnn::FloatType::F16;

  vkcnn::shaders::ActivationShader activationShader{};

  vkcnn::ActivationHostTensor inputHost{
      {vkcnn::ActivationShape{W, H, C}, inLayout, inType}};
  vkcnn::ActivationHostTensor outputHost{
      {vkcnn::ActivationShape{W, H, C}, outLayout, outType}};

  vkcnn::OpActivation activation{
      inLayout, inType, outLayout, outType, C, vkcnn::ActivationFunction::ReLU};

  vkcnn::ActivationShaderSource activationSrc =
      activationShader.do_specialize(activation);

  for (unsigned int h = 0; h < H; ++h) {
    for (unsigned int w = 0; w < W; ++w) {
      for (unsigned int c = 0; c < C; ++c) {
        inputHost.at(w, h, c) = static_cast<float>(c + 1);
      }
    }
  }

  // =========== RUNTIME ==============

  vkcnn::runtime::ActivationDeviceTensor inputDevice{inputHost.desc(),
                                                     deviceAlloc};

  vkcnn::runtime::ActivationDeviceTensor outputDevice{outputHost.desc(),
                                                      deviceAlloc};

  vkcnn::runtime::ActivationPipeline actiPipe{context, shaderCompiler,
                                              activationSrc};

  auto cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
  cmd->begin();

  inputDevice.upload(cmd, inputHost);
  outputDevice.zero(cmd);

  for (unsigned int i = 0; i < 100; ++i) {
    MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "activation");
    actiPipe.run(cmd, inputDevice, outputDevice);
  }

  auto download = outputDevice.download(cmd);

  cmd->end();
  queue->submit_wait(cmd);

  download.complete(outputHost);

  if (W <= 32) {
    fmt::println("OUT:");
    vkcnn::tensor_algo::printActivation(outputHost);
  }

  profiler->collect(true, false);
  fmt::println("{}", merian::Profiler::get_report_str(profiler->get_report()));
  double lat = profiler->get_report().gpu_total();
  std::size_t mem = 0;
  mem +=
      inputHost.shape().h * inputHost.shape().w * C * inputHost.type().size();
  mem += outputHost.shape().h * outputHost.shape().w * C *
         outputHost.type().size();

  double throughput = mem / (lat * 1e-3);
  fmt::println("Throughput: {}GB/s", throughput * 1e-9);
}

void upsample_sandbox() {
  merian::ContextHandle context = vkcnn::merian::createContext();

  merian::ShaderCompilerHandle shaderCompiler =
      std::make_shared<merian::SystemGlslcCompiler>(context);
  auto resources = context->get_extension<merian::ExtensionResources>();
  merian::ResourceAllocatorHandle deviceAlloc = resources->resource_allocator();

  merian::QueueHandle queue = context->get_queue_GCT();
  merian::CommandPoolHandle cmdPool =
      std::make_shared<merian::CommandPool>(queue);

  merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context);
  merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
      std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context);
  query_pool->reset();
  profiler->set_query_pool(query_pool);

  const unsigned int scalingFactor = 2;

  const unsigned int W = 1920 / scalingFactor;
  const unsigned int H = 1080 / scalingFactor;

  const unsigned int C = 32;

  const unsigned int W2 = W * scalingFactor;
  const unsigned int H2 = H * scalingFactor;

  const vkcnn::ActivationLayout inLayout = vkcnn::ActivationLayout::CHWC8;
  const vkcnn::ActivationLayout outLayout = vkcnn::ActivationLayout::CHWC8;

  const vkcnn::FloatType outType = vkcnn::FloatType::F16;
  const vkcnn::FloatType inType = vkcnn::FloatType::F16;

  vkcnn::shaders::DirectUpsampleShader upsampleShader{};

  vkcnn::ActivationHostTensor inputHost{
      {vkcnn::ActivationShape{W, H, C}, inLayout, inType}};
  vkcnn::ActivationHostTensor outputHost{
      {vkcnn::ActivationShape{W2, H2, C}, outLayout, outType}};

  vkcnn::OpUpsample upsample{inLayout,
                             inType,
                             outLayout,
                             outType,
                             C,
                             scalingFactor,
                             vkcnn::FilterMode::Nearest};

  vkcnn::UpsampleShaderSource upsampleSrc =
      upsampleShader.do_specialize(upsample);

  for (unsigned int h = 0; h < H; ++h) {
    for (unsigned int w = 0; w < W; ++w) {
      for (unsigned int c = 0; c < C; ++c) {
        inputHost.at(w, h, c) = static_cast<float>(w + 1 + h + 1);
      }
    }
  }

  // =========== RUNTIME ==============

  vkcnn::runtime::ActivationDeviceTensor inputDevice{inputHost.desc(),
                                                     deviceAlloc};

  vkcnn::runtime::ActivationDeviceTensor outputDevice{outputHost.desc(),
                                                      deviceAlloc};

  vkcnn::runtime::UpsamplePipeline upsamplePipe{context, shaderCompiler,
                                                upsampleSrc};

  auto cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
  cmd->begin();

  inputDevice.upload(cmd, inputHost);
  outputDevice.zero(cmd);

  for (unsigned int i = 0; i < 1; ++i) {
    MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "upsample");
    upsamplePipe.run(cmd, inputDevice, outputDevice);
  }

  auto download = outputDevice.download(cmd);

  cmd->end();
  queue->submit_wait(cmd);

  download.complete(outputHost);

  if (W2 <= 32 || false) {
    vkcnn::tensor_algo::printActivation(outputHost);
  }

  profiler->collect(true, false);
  fmt::println("{}", merian::Profiler::get_report_str(profiler->get_report()));
  double lat = profiler->get_report().gpu_total();
  std::size_t mem = 0;
  mem +=
      inputHost.shape().h * inputHost.shape().w * C * inputHost.type().size();
  mem += outputHost.shape().h * outputHost.shape().w * C *
         outputHost.type().size();

  double throughput = mem / (lat * 1e-3);
  fmt::println("Throughput: {}GB/s", throughput * 1e-9);
}

void pool_sandbox() {
  merian::ContextHandle context = vkcnn::merian::createContext();

  merian::ShaderCompilerHandle shaderCompiler =
      std::make_shared<merian::SystemGlslcCompiler>(context);
  auto resources = context->get_extension<merian::ExtensionResources>();
  merian::ResourceAllocatorHandle deviceAlloc = resources->resource_allocator();

  merian::QueueHandle queue = context->get_queue_GCT();
  merian::CommandPoolHandle cmdPool =
      std::make_shared<merian::CommandPool>(queue);

  merian::ProfilerHandle profiler = std::make_shared<merian::Profiler>(context);
  merian::QueryPoolHandle<vk::QueryType::eTimestamp> query_pool =
      std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context);
  query_pool->reset();
  profiler->set_query_pool(query_pool);

  glm::uvec2 kernelSize = glm::uvec2(2, 2);

  const unsigned int IN_W = 1920;
  const unsigned int IN_H = 1080;

  const unsigned int C = 32;

  const unsigned int OUT_W = IN_W / kernelSize.x;
  const unsigned int OUT_H = IN_H / kernelSize.y;

  const vkcnn::PoolFunction poolFunc = vkcnn::PoolFunction::Max;

  const vkcnn::ActivationLayout inLayout = vkcnn::ActivationLayout::CHWC8;
  const vkcnn::ActivationLayout outLayout = vkcnn::ActivationLayout::CHWC8;

  const vkcnn::FloatType outType = vkcnn::FloatType::F16;
  const vkcnn::FloatType inType = vkcnn::FloatType::F16;

  vkcnn::shaders::DirectPoolShader poolShader{};

  vkcnn::ActivationHostTensor inputHost{
      {vkcnn::ActivationShape{IN_W, IN_H, C}, inLayout, inType}};
  vkcnn::ActivationHostTensor outputHost{
      {vkcnn::ActivationShape{OUT_W, OUT_H, C}, outLayout, outType}};

  vkcnn::OpPool pool{
      inLayout,   inType,     outLayout,        outType,  C,
      kernelSize, kernelSize, glm::uvec2(0, 0), poolFunc,
  };

  vkcnn::PoolShaderSource poolSrc = poolShader.do_specialize(pool);

  for (unsigned int h = 0; h < IN_H; ++h) {
    for (unsigned int w = 0; w < IN_W; ++w) {
      for (unsigned int c = 0; c < C; ++c) {
        inputHost.at(w, h, c) = static_cast<float>(w + 1 + h + 1);
      }
    }
  }

  // =========== RUNTIME ==============

  vkcnn::runtime::ActivationDeviceTensor inputDevice{inputHost.desc(),
                                                     deviceAlloc};

  vkcnn::runtime::ActivationDeviceTensor outputDevice{outputHost.desc(),
                                                      deviceAlloc};

  vkcnn::runtime::PoolPipeline poolPipe{context, shaderCompiler, poolSrc};

  auto cmd = std::make_shared<merian::CommandBuffer>(cmdPool);
  cmd->begin();

  inputDevice.upload(cmd, inputHost);
  outputDevice.zero(cmd);

  for (unsigned int i = 0; i < 1; ++i) {
    MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "upsample");
    poolPipe.run(cmd, inputDevice, outputDevice);
  }

  auto download = outputDevice.download(cmd);

  cmd->end();
  queue->submit_wait(cmd);

  download.complete(outputHost);

  if (OUT_W <= 32 || false) {
    vkcnn::tensor_algo::printActivation(inputHost);

    fmt::println("Output:");

    vkcnn::tensor_algo::printActivation(outputHost);
  }

  profiler->collect(true, false);
  fmt::println("{}", merian::Profiler::get_report_str(profiler->get_report()));
  double lat = profiler->get_report().gpu_total();
  std::size_t mem = 0;
  mem +=
      inputHost.shape().h * inputHost.shape().w * C * inputHost.type().size();
  mem += outputHost.shape().h * outputHost.shape().w * C *
         outputHost.type().size();

  double throughput = mem / (lat * 1e-3);
  fmt::println("Throughput: {}GB/s", throughput * 1e-9);
}

void sym_expr_sandbox() {
  vkcnn::SymGraph g;
  auto A = g.createParameter(), B = g.createParameter();
  auto X = g.mul(g.add(A, 1), g.add(B, 2));
  const unsigned m = 5;

  auto lhs = X;
  auto rhs = g.add(g.mul(m, g.div(X, m)), g.mod(X, m));

  g.debug();

  // if (!g.resolve(lhs).isSymbolic()) {
  //   fmt::println("lhs = {}", lhs.value());
  // }
  // if (!g.resolve(rhs).isSymbolic()) {
  //   fmt::println("rhs = {}", rhs.value());
  // }
}

int main() {
  // conv_sandbox();
  // copy_transform_sandbox();
  // activation_sandbox();
  // upsample_sandbox();
  // pool_sandbox();
  sym_expr_sandbox();
  return 0;
}
