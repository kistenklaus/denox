#include "denox/compiler/compile_symbols/compile_symbols.hpp"
#include "denox/common/PushConstant.hpp"
#include "denox/compiler/frontend/model/NamedValue.hpp"
#include "denox/memory/container/dynamic_bitset.hpp"
#include <fmt/format.h>

namespace denox::compiler {

SymProgram compile_symbols(SpvSchedule &schedule, const Model &model,
                           const CompileOptions &options,
                           diag::Logger &logger) {

  logger.info("[ 98%] {}Building SymIR{}", logger.green(), logger.reset());

  memory::vector<Sym::symbol> symbols;
  memory::dynamic_bitset symbolAdded(schedule.symGraph.symbolCount());

  // const auto &inputs = schedule.inputs;
  // const auto &outputs = schedule.outputs;

  // const auto is_input = [&](uint64_t tid) -> bool {
  //   return std::ranges::find(inputs, tid) != inputs.end();
  // };
  // const auto is_output = [&](uint64_t tid) -> bool {
  //   return std::ranges::find(outputs, tid) != outputs.end();
  // };
  // const auto is_interface = [&](uint64_t tid) {
  //   return is_input(tid) || is_output(tid);
  // };

  const auto require_symbol = [&](Sym sym) -> Sym {
    Sym s = schedule.symGraph.resolve(sym);
    if (s.isConstant()) {
      return s;
    }
    if (s.isSymbolic() && !symbolAdded[s.sym()]) {
      symbols.push_back(s.sym());
      symbolAdded[s.sym()] = true;
    }
    return s;
  };

  memory::vector<NamedValue> namedValues(model.valueNames().begin(),
                                         model.valueNames().end());

  for (auto &namedValue : namedValues) {
    namedValue.value = require_symbol(namedValue.value);
  }

  for (size_t tid = 0; tid < schedule.tensors.size(); ++tid) {
    auto &view = schedule.tensors[tid];
    view.size = require_symbol(view.size);
    view.offset = require_symbol(view.offset);
    if (view.info.width) {
      view.info.width = require_symbol(*view.info.width);
    }
    if (view.info.height) {
      view.info.height = require_symbol(*view.info.height);
    }
    if (view.info.channels) {
      view.info.channels = require_symbol(*view.info.channels);
    }
  }

  for (auto &buffer : schedule.buffers) {
    buffer.size = require_symbol(buffer.size);
  }

  for (auto &dispatch : schedule.dispatches) {
    for (auto &pc : dispatch.pushConstants) {
      if (pc.isDynamic() && !symbolAdded[pc.dynamic()]) {
        symbols.push_back(pc.dynamic());
        symbolAdded[pc.dynamic()] = true;
      }
    }
    dispatch.workgroupCountX = require_symbol(dispatch.workgroupCountX);
    dispatch.workgroupCountY = require_symbol(dispatch.workgroupCountY);
    dispatch.workgroupCountZ = require_symbol(dispatch.workgroupCountZ);

    if (dispatch.info.memoryReads.has_value()) {
      dispatch.info.memoryReads = require_symbol(*dispatch.info.memoryReads);
    }
    if (dispatch.info.memoryWrites.has_value()) {
      dispatch.info.memoryWrites = require_symbol(*dispatch.info.memoryWrites);
    }
    if (dispatch.info.flops.has_value()) {
      dispatch.info.flops = require_symbol(*dispatch.info.flops);
    }
  }

  auto [ir, remap] = schedule.symGraph.compile(symbols);

  for (auto &dispatch : schedule.dispatches) {
    for (auto &pc : dispatch.pushConstants) {
      if (pc.isDynamic()) {
        pc = PushConstant(remap[pc.dynamic()], pc.type());
      }
    }
    dispatch.workgroupCountX = remap[dispatch.workgroupCountX];
    dispatch.workgroupCountY = remap[dispatch.workgroupCountY];
    dispatch.workgroupCountZ = remap[dispatch.workgroupCountZ];

    if (dispatch.info.memoryReads.has_value()) {
      dispatch.info.memoryReads = remap[*dispatch.info.memoryReads];
    }
    if (dispatch.info.memoryWrites.has_value()) {
      dispatch.info.memoryWrites = remap[*dispatch.info.memoryWrites];
    }
    if (dispatch.info.flops.has_value()) {
      dispatch.info.flops = remap[*dispatch.info.flops];
    }
  }

  for (auto &buffer : schedule.buffers) {
    buffer.size = remap[buffer.size];
  }

  for (size_t tid = 0; tid < schedule.tensors.size(); ++tid) {
    auto &view = schedule.tensors[tid];
    view.size = remap[view.size];
    view.offset = remap[view.offset];
    if (view.info.width) {
      view.info.width = remap[*view.info.width];
    }
    if (view.info.height) {
      view.info.height = remap[*view.info.height];
    }
    if (view.info.channels) {
      view.info.channels = remap[*view.info.channels];
    }
  }

  for (size_t i = 0; i < namedValues.size(); ++i) {
    namedValues[i].value = remap[namedValues[i].value];
  }

  return SymProgram{
      .namedValues = std::move(namedValues),
      .ir = std::move(ir),
  };
}

} // namespace denox::compiler
