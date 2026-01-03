#pragma once

#include "denox/compiler/implement/ComputeDispatch.hpp"
#include "denox/compiler/placement/Buffer.hpp"
#include "denox/compiler/placement/TensorInitalizer.hpp"
#include "denox/compiler/placement/TensorView.hpp"
#include "denox/memory/container/dynamic_bitset.hpp"
#include "denox/symbolic/SymGraph.hpp"
#include <cstddef>
#include <cstdint>

namespace denox::compiler {

struct MemSchedule {
  SymGraph symGraph;

  memory::vector<TensorView> tensors;
  memory::vector<Buffer> buffers;
  memory::vector<TensorInitializer> initializers;

  memory::vector<ComputeDispatch> dispatches;

  // indexes into tensors.
  memory::small_vector<uint64_t, 2> inputs;
  memory::small_vector<uint64_t, 2> outputs;
};

} // namespace denox::compiler

template <> struct fmt::formatter<denox::compiler::MemSchedule> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::compiler::MemSchedule &ms,
              FormatContext &ctx) const {
    auto out = ctx.out();

    // Mark which tensor views are backed by initializers (i.e., parameters).
    denox::memory::dynamic_bitset is_parameter_view(ms.tensors.size(), false);
    for (const auto &init : ms.initializers) {
      const std::size_t vid = static_cast<std::size_t>(init.tensor);
      if (vid < ms.tensors.size()) {
        is_parameter_view[vid] = true;
      }
    }

    fmt::format_to(out, "MemSchedule {{\n");

    // ============================================================
    // Buffers (size + alignment only)
    // ============================================================

    fmt::format_to(out, "  buffers:\n");
    for (std::size_t i = 0; i < ms.buffers.size(); ++i) {
      const auto &b = ms.buffers[i];
      fmt::format_to(out, "    [{}] = {{ size={}, alignment={} }}\n", i, b.size,
                     b.alignment);
    }

    // ============================================================
    // Tensors / Views (buffer, offset, size, info, parameter flag)
    // ============================================================

    fmt::format_to(out, "  tensors:\n");
    for (std::size_t i = 0; i < ms.tensors.size(); ++i) {
      const auto &v = ms.tensors[i];
      const bool isParam = is_parameter_view[i];

      // Print required fields + buffer (practically necessary for
      // interpretation).
      if (isParam) {
        fmt::format_to(out,
                       "    [{}] = {{ buffer={}, offset={}, size={}, info={}, "
                       "parameter=true }}\n",
                       i, v.buffer, v.offset, v.size, v.info);
      } else {
        fmt::format_to(out,
                       "    [{}] = {{ buffer={}, offset={}, size={}, info={}, "
                       "parameter=false }}\n",
                       i, v.buffer, v.offset, v.size, v.info);
      }
    }

    // ============================================================
    // Dispatches
    // ============================================================

    fmt::format_to(out, "  dispatches:\n");
    for (std::size_t i = 0; i < ms.dispatches.size(); ++i) {
      fmt::format_to(out, "    [{}] = {}\n", i, ms.dispatches[i]);
    }

    // ============================================================
    // Inputs / Outputs (indices into tensors)
    // ============================================================

    fmt::format_to(out, "  inputs: [");
    {
      bool first = true;
      for (std::uint64_t vid : ms.inputs) {
        if (!first) {
          fmt::format_to(out, ", ");
        }
        first = false;
        fmt::format_to(out, "V{}", vid);
      }
    }
    fmt::format_to(out, "]\n");

    fmt::format_to(out, "  outputs: [");
    {
      bool first = true;
      for (std::uint64_t vid : ms.outputs) {
        if (!first) {
          fmt::format_to(out, ", ");
        }
        first = false;
        fmt::format_to(out, "V{}", vid);
      }
    }
    fmt::format_to(out, "]\n");

    fmt::format_to(out, "}}");
    return out;
  }
};
