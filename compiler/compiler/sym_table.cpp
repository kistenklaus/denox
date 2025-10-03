#include "compiler/sym_table.hpp"
#include "model/DynamicInputExtent.hpp"

namespace denox::compiler {

SymTable sym_table(const Model &model, const Options &options) {
  SymTable table;

  NamedExtent inputExtentNames = model.getInputExtentNames();
  if (inputExtentNames.channels.has_value()) {
    fmt::println("IN-CH: {}", inputExtentNames.channels.value());
  } else {
    fmt::println("IN-CH: ?");
  }

  if (inputExtentNames.width.has_value()) {
    fmt::println("IN-W: {}", inputExtentNames.width.value());
  } else {
    fmt::println("IN-W: ?");
  }

  if (inputExtentNames.height.has_value()) {
    fmt::println("IN-H: {}", inputExtentNames.height.value());
  } else {
    fmt::println("IN-H: ?");
  }

  NamedExtent outputExtentNames = model.getOutputExtentNames();
  if (outputExtentNames.channels.has_value()) {
    fmt::println("OUT-CH: {}", outputExtentNames.channels.value());
  } else {
    fmt::println("OUT-CH: ?");
  }

  if (outputExtentNames.width.has_value()) {
    fmt::println("OUT-W: {}", outputExtentNames.width.value());
  } else {
    fmt::println("OUT-W: ?");
  }

  if (outputExtentNames.height.has_value()) {
    fmt::println("OUT-H: {}", outputExtentNames.height.value());
  } else {
    fmt::println("OUT-H: ?");
  }

  return table;
}

} // namespace denox::compiler
