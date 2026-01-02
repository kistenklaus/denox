#pragma once

#include "denox/compiler/implement/ComputeDispatch.hpp"
#include "denox/compiler/implement/MemoryConstrain.hpp"
#include "denox/compiler/implement/Parameter.hpp"
#include "denox/compiler/implement/Tensor.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/symbolic/SymGraph.hpp"
#include <fmt/format.h>
namespace denox::compiler {

struct OptSchedule {
  SymGraph symGraph;
  memory::vector<Tensor> tensors;
  memory::vector<ComputeDispatch> dispatches;
  memory::vector<MemoryImplicitConcatConstrain> memoryConstrains;
  memory::vector<Parameter> parameters;

  memory::vector<uint64_t> inputs;
  memory::vector<uint64_t> outputs;
};

} // namespace denox::compiler

template <> struct fmt::formatter<denox::compiler::OptSchedule> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::compiler::OptSchedule &s, FormatContext &ctx) const {
    auto out = ctx.out();

    auto print_indexed = [&](auto &&vec, const char *name) {
      fmt::format_to(out, "-{}:\n", name);
      for (std::size_t i = 0; i < vec.size(); ++i) {
        fmt::format_to(out, "[{}] = {}\n", i, vec[i]);
      }
    };

    fmt::format_to(out, "{{\n");

    print_indexed(s.tensors, "tensors");
    print_indexed(s.dispatches, "dispatches");
    print_indexed(s.parameters, "parameters");

    fmt::format_to(out,
                   "-memoryConstrains: {},\n"
                   "-inputs: {},\n"
                   "-outputs: {}\n"
                   "}}",
                   s.memoryConstrains, s.inputs, s.outputs);

    return out;
  }
};
