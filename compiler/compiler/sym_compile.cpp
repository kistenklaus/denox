#include "compiler/sym_compile.hpp"
#include "memory/container/dynamic_bitset.hpp"
#include "symbolic/SymGraph.hpp"

namespace denox::compiler {

SymIR sym_compile(const CompModel &model) {

  memory::vector<Sym::symbol> symbols;
  memory::dynamic_bitset symbolAdded(model.symGraph.symbolCount(), false);

  for (const auto &dispatch : model.dispatches) {
    for (const auto &pushConstant : dispatch.pushConstants) {
      // TODO
    }
  }

  for (const auto &buffer : model.buffers) {
    if (buffer.size.isSymbolic() && !symbolAdded[buffer.size.sym()]) {
      symbols.push_back(buffer.size.sym());
      symbolAdded[buffer.size.sym()] = true;
    }
  }
  for (const auto &tensor : model.tensors) {
    if (tensor.offset.isSymbolic() && !symbolAdded[tensor.offset.sym()]) {
      symbols.push_back(tensor.offset.sym());
      symbolAdded[tensor.offset.sym()] = true;
    }
  }

  const auto [symIR, remap] = model.symGraph.compile(symbols);
  return symIR;
}

} // namespace denox::compiler
