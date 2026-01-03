#include "denox/compiler/frontend/frontend.hpp"
#include "denox/common/TensorFormat.hpp"
#include "denox/compiler/Options.hpp"
#include "denox/compiler/frontend/onnx/onnx.hpp"
#include <algorithm>

denox::compiler::Model
denox::compiler::frontend(memory::span<const std::byte> raw,
                          const CompileOptions &options) {
  Model model = denox::onnx::read(raw, options);

  const auto inputNames = model.getInputNames();
  for (const auto &inputName : inputNames) {
    auto input = *model.getInput(inputName);

    auto interfaceDescriptor =
        std::ranges::find_if(options.interfaceDescriptors,
                             [&](const InterfaceTensorDescriptor &descriptor) {
                               return descriptor.name == inputName;
                             });
    if (interfaceDescriptor != options.interfaceDescriptors.end()) {
      if (interfaceDescriptor->storage != TensorStorage::Optimal) {
        input.setStorage(interfaceDescriptor->storage);
      }
      if (interfaceDescriptor->format != TensorFormat::Optimal) {
        input.setFormat(interfaceDescriptor->format);
      }
    }
  }

  const auto outputNames = model.getOutputNames();
  for (const auto &outputName : outputNames) {
    auto output = *model.getOutput(outputName);

    auto interfaceDescriptor =
        std::ranges::find_if(options.interfaceDescriptors,
                             [&](const InterfaceTensorDescriptor &descriptor) {
                               return descriptor.name == outputName;
                             });
    if (interfaceDescriptor != options.interfaceDescriptors.end()) {
      if (interfaceDescriptor->storage != TensorStorage::Optimal) {
        output.setStorage(interfaceDescriptor->storage);
      }
      if (interfaceDescriptor->format != TensorFormat::Optimal) {
        output.setFormat(interfaceDescriptor->format);
      }
    }
  }

  return model;
}
