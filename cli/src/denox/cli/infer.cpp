#include "denox/cli/infer.hpp"
#include "denox/cli/io/InputStream.hpp"
#include "denox/cli/io/OutputStream.hpp"
#include "denox/cli/npy/NpyInputStream.hpp"
#include "denox/cli/npy/NpyOutputStream.hpp"
#include "denox/cli/png/PngInputStream.hpp"
#include "denox/cli/png/PngOutputStream.hpp"
#include "denox/common/TensorDataType.hpp"
#include "denox/common/TensorStorage.hpp"
#include "denox/compiler/compile.hpp"
#include "denox/device_info/query/query_driver_device_info.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/runtime/instance.hpp"
#include "denox/symbolic/SymGraphEval.hpp"
#include <fmt/ostream.h>

void infer(InferAction &action) {
  const char *deviceName = nullptr;
  if (std::holds_alternative<IOEndpoint>(action.device)) {
    throw std::runtime_error("invalid device");
  } else if (std::holds_alternative<denox::memory::string>(action.device)) {
    deviceName = std::get<denox::memory::string>(action.device).c_str();
  }

  denox::diag::Logger logger("denox.infer", action.logcolors, action.loglevel);

  const auto ctx =
      denox::runtime::Context::make(deviceName, action.apiVersion, logger);
  denox::runtime::ModelHandle model;

  switch (action.model.kind()) {
  case ArtefactKind::Onnx: {
    denox::memory::optional<denox::Db> db;
    if (action.database) {
      db = denox::Db::open(action.database->endpoint.path());
    }
    denox::runtime::ContextHandle context =
        denox::runtime::Context::make(deviceName, action.apiVersion, logger);

    action.options.deviceInfo = denox::query_driver_device_info(
        vk::Instance{context->vkInstance()},
        vk::PhysicalDevice{context->vkPhysicalDevice()}, action.apiVersion);
    auto dnxbuf = denox::compile(action.model.dnx().data, db, context,
                                 action.options, logger);
    model = denox::runtime::Model::make(dnxbuf, ctx);
    break;
  }
  case ArtefactKind::Dnx: {
    model = denox::runtime::Model::make(action.model.dnx().data, ctx);
    break;
  }
  case ArtefactKind::Database:
    denox::diag::invalid_state();
  }

  IOEndpoint input = action.input;
  IOEndpoint output = action.output;

  bool outputPng = false;
  if (output.kind() == IOEndpointKind::Path &&
      output.path().extension() == ".png") {
    outputPng = true;
  }
  bool outputNpy = false;
  if (output.kind() == IOEndpointKind::Path &&
      output.path().extension() == ".npy") {
    outputNpy = true;
  }

  InputStream inputStream(input);

  OutputStream outputStream(output);

  assert(model->inputs().size() == 1);
  assert(model->outputs().size() == 1);

  const auto &modelInput = model->tensors()[model->inputs().front()];
  const denox::TensorFormat inputFormat = *modelInput.format;
  denox::memory::ActivationLayout inputLayout =
      denox::memory::ActivationLayout::HWC;
  switch (inputFormat) {
  case denox::TensorFormat::Optimal:
    throw std::runtime_error("invalid input format");
  case denox::TensorFormat::SSBO_HWC:
    inputLayout = denox::memory::ActivationLayout::HWC;
    break;
  case denox::TensorFormat::SSBO_CHW:
    inputLayout = denox::memory::ActivationLayout::CHW;
    break;
  case denox::TensorFormat::SSBO_CHWC8:
    inputLayout = denox::memory::ActivationLayout::CHWC8;
    break;
  case denox::TensorFormat::TEX_RGBA:
  case denox::TensorFormat::TEX_RGB:
  case denox::TensorFormat::TEX_RG:
  case denox::TensorFormat::TEX_R:
    throw std::runtime_error("texture formats are not supported!");
  }

  const denox::TensorDataType inputTensorDtype = *modelInput.dtype;
  denox::memory::Dtype inputDtype;
  switch (inputTensorDtype) {
  case denox::TensorDataType::Auto:
    throw std::runtime_error("invalid input dtype");
  case denox::TensorDataType::Float16:
    inputDtype = denox::memory::Dtype::F16;
    break;
  case denox::TensorDataType::Float32:
    inputDtype = denox::memory::Dtype::F32;
    break;
  case denox::TensorDataType::Float64:
    inputDtype = denox::memory::Dtype::F64;
    break;
  }

  struct InstanceCache {
    denox::memory::vector<denox::SymSpec> spec;
    denox::runtime::InstanceHandle instance;
  };
  denox::memory::optional<InstanceCache> instanceCache;

  while (true) {
    denox::memory::optional<denox::memory::ActivationTensor> parsed;
    const auto prefix = inputStream.peek(8);
    if (prefix.empty()) {
      break; // EOF.
    }
    if (prefix.size() < 8) {
      throw std::runtime_error("Truncated input");
    }

    uint64_t magic;
    std::memcpy(&magic, prefix.data(), sizeof(magic));

    if (PngInputStream::is_png(magic)) {
      parsed =
          PngInputStream{&inputStream}.read_image();
    } else if (NpyInputStream::is_npy(magic)) {
      parsed =
          NpyInputStream{&inputStream}.read_tensor();
    } else {
      throw std::runtime_error("invalid input");
    }

    if (!parsed) {
      throw std::runtime_error("Decoder returned EOF after a valid signature");
    }

    const denox::memory::ActivationTensor &inTensor = *parsed;

    denox::memory::ActivationDescriptor desc{
        {inTensor.shape().w, inTensor.shape().h, inTensor.shape().c},
        inputLayout,
        inputDtype,
    };

    const denox::memory::ActivationTensor tensor{desc, inTensor};

    auto input = model->tensors()[model->inputs().front()];
    assert(input.width.has_value());
    assert(input.height.has_value());
    assert(input.channels.has_value());

    denox::memory::vector<denox::SymSpec> specs;

    if (input.width->isSymbolic()) {
      specs.emplace_back(input.width->sym(), tensor.shape().w);
    } else {
      uint64_t expected = static_cast<uint64_t>(input.width->constant());
      if (expected != tensor.shape().w) {
        throw std::runtime_error(
            fmt::format("input has invalid width. Expected {}, got {}",
                        expected, tensor.shape().w));
      }
    }
    if (input.height->isSymbolic()) {
      specs.emplace_back(input.height->sym(), tensor.shape().h);
    } else {
      uint64_t expected = static_cast<uint64_t>(input.height->constant());
      if (expected != tensor.shape().h) {
        throw std::runtime_error(
            fmt::format("input has invalid height. Expected {}, got {}",
                        expected, tensor.shape().h));
      }
    }

    if (input.channels->isSymbolic()) {
      specs.emplace_back(input.channels->sym(), tensor.shape().c);
    } else {
      uint64_t expected = static_cast<uint64_t>(input.channels->constant());
      if (expected != tensor.shape().c) {
        throw std::runtime_error(
            fmt::format("input has invalid channel count. Expected {}, got {}",
                        expected, tensor.shape().c));
      }
    }

    denox::runtime::InstanceHandle instance;
    if (instanceCache) {
      auto cachedSpecs = instanceCache->spec;
      bool equal = cachedSpecs.size() == specs.size();
      for (size_t i = 0; i < cachedSpecs.size() && equal; ++i) {
        equal = cachedSpecs[i].symbol == specs[i].symbol &&
                cachedSpecs[i].value == specs[i].value;
      }
      if (equal) {
        instance = instanceCache->instance;
      } else {
        instance = denox::runtime::Instance::make(model, specs, logger);
        instanceCache.emplace(std::move(specs), instance);
      }
    } else {
      instance = denox::runtime::Instance::make(model, specs, logger);
      instanceCache.emplace(std::move(specs), instance);
    }

    const void *inputData = static_cast<const void *>(tensor.data());
    const void **pInput = &inputData;

    auto outdesc = instance->getOutputDesc(0);

    denox::memory::ActivationTensor outputTensor{outdesc};
    void *outputData = static_cast<void *>(outputTensor.data());
    void **pOutput = &outputData;

    instance->infer(pInput, pOutput);

    if (outputPng || (output.kind() == IOEndpointKind::Pipe &&
                      PngInputStream::is_png(magic))) {
      PngOutputStream{&outputStream}.write_image(outputTensor);
    } else if (outputNpy || (output.kind() == IOEndpointKind::Pipe &&
                             NpyInputStream::is_npy(magic))) {
      NpyOutputStream{&outputStream}.write_tensor(outputTensor);
    } else {
      throw std::runtime_error("invalid state !");
    }
  }
}
