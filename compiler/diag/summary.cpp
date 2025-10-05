#include "diag/summary.hpp"

namespace denox::compiler::diag {

void print_summary(const Model &model, const ImplModel &implModel,
                   const CompModel &compModel, const SymIR &symIR,
                   std::size_t symCount,
                   const flatbuffers::DetachedBuffer &dnx) {
  fmt::println("\n\x1B[31mSummary:\x1B[0m");
  auto inputExtentNames = model.getInputExtentNames();
  fmt::println(
      "\u2022 {:<20} : [{}, {}, {}]", "Input-Shape (HWC) ",
      model.getInput().height().isSymbolic()
          ? (inputExtentNames.height.has_value()
                 ? inputExtentNames.height.value()
                 : "?")
          : (inputExtentNames.height.has_value()
                 ? fmt::format("{}={}", (inputExtentNames.height.value()),
                               model.getInput().height().constant())
                 : fmt::format("{}", model.getInput().height().constant())),
      model.getInput().width().isSymbolic()
          ? (inputExtentNames.width.has_value() ? inputExtentNames.width.value()
                                                : "?")
          : (inputExtentNames.width.has_value()
                 ? fmt::format("{}={}", (inputExtentNames.width.value()),
                               model.getInput().width().constant())
                 : fmt::format("{}", model.getInput().width().constant())),
      inputExtentNames.channels.has_value()
          ? fmt::format("{}={}", inputExtentNames.channels.value(),
                        model.getInput().channels())
          : fmt::format("{}", model.getInput().channels()));

  auto outputExtentNames = model.getOutputExtentNames();
  fmt::println(
      "\u2022 {:<20} : [{}, {}, {}]", "Output-Shape (HWC) ",
      model.getOutput().height().isSymbolic()
          ? (outputExtentNames.height.has_value()
                 ? outputExtentNames.height.value()
                 : "?")
          : (outputExtentNames.height.has_value()
                 ? fmt::format("{}={}", (outputExtentNames.height.value()),
                               model.getOutput().height().constant())
                 : fmt::format("{}", model.getOutput().height().constant())),
      model.getOutput().width().isSymbolic()
          ? (outputExtentNames.width.has_value()
                 ? outputExtentNames.width.value()
                 : "?")
          : (outputExtentNames.width.has_value()
                 ? fmt::format("{}={}", (outputExtentNames.width.value()),
                               model.getOutput().width().constant())
                 : fmt::format("{}", model.getOutput().width().constant())),
      outputExtentNames.channels.has_value()
          ? fmt::format("{}={}", outputExtentNames.channels.value(),
                        model.getOutput().channels())
          : fmt::format("{}", model.getOutput().channels()));

  fmt::println("\u2022 {:<20} : {}", "Outputs ", compModel.outputs.size());

  fmt::println("\u2022 {:<20} : {}", "Number-Of-Dispatches",
               implModel.dispatches.size());

  std::size_t spirvByteSize = 0;
  for (std::size_t d = 0; d < compModel.dispatches.size(); ++d) {
    spirvByteSize += compModel.dispatches[d].src.size;
  }
  if (spirvByteSize > 1000000) {
    fmt::println("\u2022 {:<20} : {:.1f}MB", "SPIRV-ByteSize",
                 static_cast<float>(spirvByteSize) / 1000000.0f);
  } else if (spirvByteSize > 1000) {
    fmt::println("\u2022 {:<20} : {:.1f}KB", "SPIRV-ByteSize",
                 static_cast<float>(spirvByteSize) / 1000.0f);
  } else {
    fmt::println("\u2022 {:<20} : {}B", "SPIRV-ByteSize", spirvByteSize);
  }

  std::size_t parameterByteSize = 0;
  for (std::size_t p = 0; p < implModel.parameters.size(); ++p) {
    parameterByteSize += implModel.parameters[p].data.size();
  }
  if (parameterByteSize > 1000000) {
    fmt::println("\u2022 {:<20} : {:.1f}MB", "Parameter-ByteSize",
                 static_cast<float>(parameterByteSize) / 1000000.0f);
  } else if (parameterByteSize > 1000) {
    fmt::println("\u2022 {:<20} : {:.1f}KB", "Parameter-ByteSize",
                 static_cast<float>(parameterByteSize) / 1000.0f);
  } else {
    fmt::println("\u2022 {:<20} : {}B", "Parameter-ByteSize",
                 parameterByteSize);
  }
  std::size_t dnxByteSize = dnx.size();
  if (dnxByteSize > 1000000) {
    fmt::println("\u2022 {:<20} : {:.1f}MB", "DNX-ByteSize",
                 static_cast<float>(dnxByteSize) / 1000000.0f);
  } else if (dnxByteSize > 1000) {
    fmt::println("\u2022 {:<20} : {:.1f}KB", "DNX-ByteSize",
                 static_cast<float>(dnxByteSize) / 1000.0f);
  } else {
    fmt::println("\u2022 {:<20} : {}B", "DNX-ByteSize", dnxByteSize);
  }

  fmt::println("\u2022 {:<20} : {}", "Dynamic Variables", symIR.varCount);
  fmt::println("\u2022 {:<20} : {}", "Dynamic Expressions", symCount);
  fmt::println("\u2022 {:<20} : {}", "SymIR OpCount", symIR.ops.size());
  fmt::println("\u2022 {:<20} : {}", "Amount of buffers",
               compModel.buffers.size());
  fmt::println("\u2022 {:<20} : {}", "Tensor views ", compModel.tensors.size());
}
} // namespace denox::compiler::diag
