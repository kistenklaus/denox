#include "compiler/sym_compile.hpp"
#include "denox/diag/not_implemented.hpp"
#include "denox/memory/container/dynamic_bitset.hpp"
#include "denox/symbolic/SymGraph.hpp"

namespace denox::compiler {

std::pair<SymIR, std::uint32_t> compile_sym_and_remap(CompModel &model,
                                                      SymTable &symTable) {

  // model.symGraph.debugDump();

  // resolve all symbols.
  for (auto &[sym, name] : symTable.symbolNames) {
    sym = model.symGraph.resolve(sym);
  }
  for (auto &buffer : model.buffers) {
    buffer.size = model.symGraph.resolve(buffer.size);
  }

  for (auto &tensor : model.tensors) {
    tensor.offset = model.symGraph.resolve(tensor.offset);
    tensor.size = model.symGraph.resolve(tensor.size);
  }
  for (auto &input : model.inputs) {
    input.extent.x = sym(model.symGraph.resolve(input.extent.x.asSym()));
    input.extent.y = sym(model.symGraph.resolve(input.extent.y.asSym()));
  }

  for (auto &output : model.outputs) {
    output.extent.x = sym(model.symGraph.resolve(output.extent.x.asSym()));
    output.extent.y = sym(model.symGraph.resolve(output.extent.y.asSym()));
  }

  memory::vector<Sym::symbol> symbols;
  memory::dynamic_bitset symbolAdded(model.symGraph.symbolCount(), false);

  for (const auto &[sym, name] : symTable.symbolNames) {
    if (sym.isSymbolic()) {
      Sym::symbol s = sym.sym();
      if (!symbolAdded[s]) {
        symbols.push_back(s);
        symbolAdded[s] = true;
      }
    }
  }

  for (const auto &dispatch : model.dispatches) {
    for (const PushConstant &pushConstant : dispatch.pushConstants) {
      if (pushConstant.isDynamic()) {
        Sym::symbol sym = pushConstant.dynamic();
        if (!symbolAdded[sym]) {
          symbols.push_back(sym);
          symbolAdded[sym] = true;
        }
      }
    }
    for (std::size_t i = 0; i < dispatch.workgroupCount.size(); ++i) {
      if (dispatch.workgroupCount[i].isSymbolic() &&
          !symbolAdded[dispatch.workgroupCount[i].sym()]) {
        symbols.push_back(dispatch.workgroupCount[i].sym());
        symbolAdded[dispatch.workgroupCount[i].sym()] = true;
      }
    }
    if (dispatch.memory_reads.has_value() && dispatch.memory_reads->isSymbolic()
        && !symbolAdded[dispatch.memory_reads->sym()]) {
      symbols.push_back(dispatch.memory_reads->sym());
      symbolAdded[dispatch.memory_reads->sym()] = true;
    } 
    if (dispatch.memory_writes.has_value() && dispatch.memory_writes->isSymbolic()
        && !symbolAdded[dispatch.memory_writes->sym()]) {
      symbols.push_back(dispatch.memory_writes->sym());
      symbolAdded[dispatch.memory_writes->sym()] = true;
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
    if (tensor.size.isSymbolic() && !symbolAdded[tensor.size.sym()]) {
      symbols.push_back(tensor.size.sym());
      symbolAdded[tensor.size.sym()] = true;
    }
  }

  for (const auto &input : model.inputs) {
    if (input.extent.x.isSymbolic() && !symbolAdded[input.extent.x.symbol()]) {
      symbols.push_back(input.extent.x.symbol());
      symbolAdded[input.extent.x.symbol()] = true;
    }

    if (input.extent.y.isSymbolic() && !symbolAdded[input.extent.y.symbol()]) {
      symbols.push_back(input.extent.y.symbol());
      symbolAdded[input.extent.y.symbol()] = true;
    }
  }

  for (const auto &output : model.outputs) {
    if (output.extent.x.isSymbolic() &&
        !symbolAdded[output.extent.x.symbol()]) {
      symbols.push_back(output.extent.x.symbol());
      symbolAdded[output.extent.x.symbol()] = true;
    }

    if (output.extent.y.isSymbolic() &&
        !symbolAdded[output.extent.y.symbol()]) {
      symbols.push_back(output.extent.y.symbol());
      symbolAdded[output.extent.y.symbol()] = true;
    }
  }

  const auto [symIR, remap] = model.symGraph.compile(symbols);

  for (auto &[sym, name] : symTable.symbolNames) {
    sym = remap[sym];
  }

  for (auto &dispatch : model.dispatches) {
    for (auto &pushConstant : dispatch.pushConstants) {
      if (pushConstant.isDynamic()) {
        Sym sym = remap[pushConstant.dynamic()];
        memory::Dtype dtype = pushConstant.type();
        if (sym.isConstant()) {
          switch (dtype.kind()) {
          case memory::DtypeKind::F16:
          case memory::DtypeKind::F32:
          case memory::DtypeKind::F64:
            diag::not_implemented();
          case memory::DtypeKind::U32:
            pushConstant =
                PushConstant(static_cast<std::uint32_t>(sym.constant()));
            break;
          case memory::DtypeKind::I32:
            pushConstant =
                PushConstant(static_cast<std::int32_t>(sym.constant()));
            break;
          }
        } else {
          pushConstant =
              PushConstant::Dynamic(remap[pushConstant.dynamic()], dtype);
        }
      }
    }
    dispatch.workgroupCount[0] = remap[dispatch.workgroupCount[0]];
    dispatch.workgroupCount[1] = remap[dispatch.workgroupCount[1]];
    dispatch.workgroupCount[2] = remap[dispatch.workgroupCount[2]];

    if (dispatch.memory_reads) {
      dispatch.memory_reads = remap[*dispatch.memory_reads];
    }

    if (dispatch.memory_writes) {
      dispatch.memory_writes = remap[*dispatch.memory_writes];
    }
  }

  for (auto &buffer : model.buffers) {
    buffer.size = remap[buffer.size];
  }
  for (auto &tensor : model.tensors) {
    tensor.offset = remap[tensor.offset];
    tensor.size = remap[tensor.size];
  }

  for (auto &input : model.inputs) {
    input.extent.x = sym(remap[input.extent.x.asSym()]);
    input.extent.y = sym(remap[input.extent.y.asSym()]);
  }
  for (auto &output : model.outputs) {
    output.extent.x = sym(remap[output.extent.x.asSym()]);
    output.extent.y = sym(remap[output.extent.y.asSym()]);
  }

  std::size_t symCount = 0;
  for (auto s : symbols) {
    Sym sym = remap[s];
    if (sym.isSymbolic()) {
      symCount += 1;
    }
  }

  return std::make_pair(symIR, symCount);
}

} // namespace denox::compiler
