#include "compiler/sym_table.hpp"
#include "model/DynamicInputExtent.hpp"

namespace denox::compiler {

SymTable sym_table(const Model &model, [[maybe_unused]] const Options &options) {
  SymTable table;

  NamedExtent inputExtentNames = model.getInputExtentNames();
  if (inputExtentNames.channels.has_value()) {
    table.symbolNames.emplace_back(Sym::Const(model.getInput().channels()), inputExtentNames.channels.value());
  } 

  if (inputExtentNames.width.has_value()) {
    table.symbolNames.emplace_back(model.getInput().width().resolve(), inputExtentNames.width.value());
  } 

  if (inputExtentNames.height.has_value()) {
    table.symbolNames.emplace_back(model.getInput().height().resolve(), inputExtentNames.height.value());
  } 

  NamedExtent outputExtentNames = model.getOutputExtentNames();
  if (outputExtentNames.channels.has_value()) {
    table.symbolNames.emplace_back(Sym::Const(model.getOutput().channels()), outputExtentNames.channels.value());
  } 

  if (outputExtentNames.width.has_value()) {
    table.symbolNames.emplace_back(model.getOutput().width().resolve(), outputExtentNames.width.value());
  } 

  if (outputExtentNames.height.has_value()) {
    table.symbolNames.emplace_back(model.getOutput().height().resolve(), outputExtentNames.height.value());
  } 

  return table;
}

} // namespace denox::compiler
