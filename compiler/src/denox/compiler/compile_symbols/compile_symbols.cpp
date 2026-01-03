#include "denox/compiler/compile_symbols/compile_symbols.hpp"
#include "denox/compiler/frontend/model/NamedValue.hpp"
#include "denox/compiler/implement/PushConstant.hpp"
#include "denox/memory/container/dynamic_bitset.hpp"
#include <algorithm>
#include <fmt/base.h>
#include <fmt/ostream.h>

namespace denox::compiler {

SymProgram compile_symbols(SpvSchedule &schedule, const Model &model,
                           const Options &options) {

  fmt::println("[ 98%] \x1B[32mBuilding SymIR of {} unique expressions\x1b[0m",
               schedule.symGraph.symbolCount());

  memory::vector<Sym::symbol> symbols;
  memory::dynamic_bitset symbolAdded(schedule.symGraph.symbolCount());

  const auto &inputs = schedule.inputs;
  const auto &outputs = schedule.outputs;

  const auto is_input = [&](uint64_t tid) -> bool {
    return std::ranges::find(inputs, tid) != inputs.end();
  };
  const auto is_output = [&](uint64_t tid) -> bool {
    return std::ranges::find(outputs, tid) != outputs.end();
  };
  const auto is_interface = [&](uint64_t tid) {
    return is_input(tid) || is_output(tid);
  };

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
    auto& view = schedule.tensors[tid];
    view.size = require_symbol(view.size);
    view.offset = require_symbol(view.offset);
    if (is_interface(tid) || options.debugInfo == DebugInfo::Enable) {
      view.info.width = require_symbol(view.info.width);
      view.info.height = require_symbol(view.info.height);
      view.info.channels = require_symbol(view.info.channels);
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

    if (options.debugInfo == DebugInfo::Enable) {
      if (dispatch.info.memoryReads.has_value()) {
        dispatch.info.memoryReads = require_symbol(*dispatch.info.memoryReads);
      }
      if (dispatch.info.memoryWrites.has_value()) {
        dispatch.info.memoryWrites =
            require_symbol(*dispatch.info.memoryWrites);
      }
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

    if (options.debugInfo == DebugInfo::Enable) {
      if (dispatch.info.memoryReads.has_value()) {
        dispatch.info.memoryReads = remap[*dispatch.info.memoryReads];
      }
      if (dispatch.info.memoryWrites.has_value()) {
        dispatch.info.memoryWrites = remap[*dispatch.info.memoryWrites];
      }
    }
  }

  for (auto &buffer : schedule.buffers) {
    buffer.size = remap[buffer.size];
  }

  for (size_t tid = 0; tid < schedule.tensors.size(); ++tid) {
    auto& view = schedule.tensors[tid];
    view.size = remap[view.size];
    view.offset = remap[view.offset];
    if (is_interface(tid) || options.debugInfo == DebugInfo::Enable) {
      view.info.width = remap[view.info.width];
      view.info.height = remap[view.info.height];
      view.info.channels = remap[view.info.channels];
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
