#include "denox/compiler/frontend/frontend.hpp"
#include "denox/compiler/frontend/onnx/onnx.hpp"

denox::compiler::Model
denox::compiler::frontend(memory::span<const std::byte> raw,
                          const Options &options) {
  auto model = denox::onnx::read(raw, options);

  // model.getInput().setLayout(options.inputLayout);
  // model.getOutput().setLayout(options.outputLayout);

  return model;
}
