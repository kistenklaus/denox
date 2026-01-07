#include "denox/cli/infer.hpp"
#include "denox/cli/io/InputStream.hpp"
#include "denox/cli/io/OutputStream.hpp"
#include "denox/cli/png/PngInputStream.hpp"
#include "denox/cli/png/PngOutputStream.hpp"
#include "denox/compiler/compile.hpp"
#include "denox/device_info/query/query_driver_device_info.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/runtime/instance.hpp"
#include <fmt/base.h>
#include <fmt/ostream.h>

void infer(InferAction &action) {

  const char *device = nullptr;
  if (action.deviceName) {
    device = action.deviceName->c_str();
  }

  const auto ctx = denox::runtime::Context::make(device, action.apiVersion);
  denox::runtime::ModelHandle model;

  switch (action.model.kind()) {
  case ArtefactKind::Onnx: {
    denox::memory::optional<denox::Db> db;
    if (action.database) {
      db = denox::Db::open(action.database->endpoint.path());
    }
    action.options.deviceInfo = denox::query_driver_device_info(
        ctx->vkInstance(), action.deviceName, action.apiVersion);
    auto dnxbuf = denox::compile(action.model.dnx().data, db, action.options);
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

  InputStream inputStream(input);
  OutputStream outputStream(output);

  assert(model->inputs().size() == 1);
  assert(model->outputs().size() == 1);

  while (!inputStream.eof()) {
    denox::memory::optional<denox::memory::ActivationTensor> parsed;
    parsed = PngInputStream{&inputStream}.read_image();
    bool isPng = parsed.has_value();

    if (!parsed) {
      // possibly fallback to raw HWC or something (unless eof ofcause)
      // TODO alternative ways of loading the tensor
    }

    fmt::println("has-image: {}", parsed.has_value());
    if (!parsed.has_value()) {
      break;
    }

    const denox::memory::ActivationTensor &image = *parsed;

    fmt::println("image-shape {}x{}x{}", image.shape().w, image.shape().h,
                 image.shape().c);

    denox::memory::ActivationDescriptor desc{
        {image.shape().w, image.shape().h, image.shape().c},
        denox::memory::ActivationLayout::HWC,
        denox::memory::Dtype::F16,
    };

    const denox::memory::ActivationTensor tensor{desc, image};
    fmt::println("input-shape {}x{}x{}", tensor.shape().w, tensor.shape().h,
                 tensor.shape().c);

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
    auto instance = denox::runtime::Instance::make(model, specs);

    const void *inputData = static_cast<const void *>(tensor.data());
    const void **pInput = &inputData;

    auto outdesc = instance->getOutputDesc(0);

    fmt::println("output-shape {}x{}x{}", outdesc.shape.w, outdesc.shape.h,
                 outdesc.shape.c);

    denox::memory::ActivationTensor output{outdesc};
    void *outputData = static_cast<void *>(output.data());
    void **pOutput = &outputData;

    instance->infer(pInput, pOutput);

    float total = 0;
    for (uint32_t h = 0; h < output.shape().h; ++h) {
      for (uint32_t w = 0; w < output.shape().w; ++w) {
        for (uint32_t c = 0; c < output.shape().c; ++c) {
          total += static_cast<float>(output.at(w, h, c));
        }
      }
    }
    fmt::println("TOTAL: {}", total);

    if (isPng) {
      PngOutputStream{&outputStream}.write_image(output);
    }
  }
}
