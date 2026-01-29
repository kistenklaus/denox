#include "denox/compiler/placement/placement.hpp"

#include "denox/algorithm/union_find.hpp"
#include "denox/diag/not_implemented.hpp"
#include "denox/diag/unreachable.hpp"
#include "denox/memory/container/dynamic_bitset.hpp"
#include "denox/memory/container/vector.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>

namespace denox::compiler {

static constexpr uint64_t u64sential = std::numeric_limits<uint64_t>::max();

MemSchedule placement(const OptSchedule &schedule) {
  MemSchedule out{};
  out.symGraph = schedule.symGraph;
  out.dispatches = schedule.dispatches;

  SymGraph &sym = out.symGraph;

  const size_t tensorCount = schedule.tensors.size();

  // Classify tensors: parameter vs runtime
  memory::dynamic_bitset isParameter(tensorCount, false);
  for (const Parameter &p : schedule.parameters) {
    const uint64_t tid = p.tensorId.index;
    assert(tid < tensorCount);
    isParameter[tid] = true;
  }

  memory::vector<uint64_t> paramTensorIds;
  memory::vector<uint64_t> runtimeTensorIds;
  paramTensorIds.reserve(tensorCount);
  runtimeTensorIds.reserve(tensorCount);

  memory::vector<uint64_t> runtimeIndexOfTensor(tensorCount, u64sential);

  for (uint64_t tid = 0; tid < tensorCount; ++tid) {
    if (isParameter[tid]) {
      paramTensorIds.push_back(tid);
    } else {
      runtimeIndexOfTensor[tid] = runtimeTensorIds.size();
      runtimeTensorIds.push_back(tid);
    }
  }

  memory::vector<uint64_t> tensorToView(tensorCount, u64sential);
  // Materialize parameter tensors: 1 tensor == 1 buffer == 1 view
  out.buffers.reserve(out.buffers.size() + paramTensorIds.size());
  out.tensors.reserve(out.tensors.size() + tensorCount);

  for (uint64_t tid : paramTensorIds) {
    const Tensor &t = schedule.tensors[tid];

    const uint64_t bufferId = out.buffers.size();
    out.buffers.push_back(Buffer{
        .size = t.size,
        .alignment = static_cast<std::uint16_t>(t.alignment),
        .first_use = std::numeric_limits<std::uint32_t>::max(), // filled later
        .last_use = 0,                                          // filled later
    });

    const uint64_t viewId = out.tensors.size();
    out.tensors.push_back(TensorView{
        .buffer = bufferId,
        .offset = Sym::Const(0),
        .size = t.size,
        .info = t.info,
    });

    tensorToView[tid] = viewId;
  }
  out.initializers.reserve(out.initializers.size() +
                           schedule.parameters.size());
  for (const Parameter &p : schedule.parameters) {
    const uint64_t tid = p.tensorId.index;
    const uint64_t viewId = tensorToView[tid];
    assert(viewId != u64sential);

    out.initializers.push_back(TensorInitializer{
        .tensor = static_cast<std::uint32_t>(viewId),
        .data = p.lazyValue(),
    });
  }

  // Alias grouping for runtime tensors via union-find (implicit concat only)
  denox::union_find<memory::vector<size_t>> uf(runtimeTensorIds.size());

  for (const MemoryImplicitConcatConstrain &c : schedule.memoryConstrains) {
    const uint64_t t0 = c.src0.index;
    const uint64_t t1 = c.src1.index;
    const uint64_t t2 = c.dst.index;

    // Current implementation: constraints involving parameters are not
    // supported.
    if (isParameter[t0] || isParameter[t1] || isParameter[t2]) {
      diag::not_implemented();
    }

    const uint64_t i0 = runtimeIndexOfTensor[t0];
    const uint64_t i1 = runtimeIndexOfTensor[t1];
    const uint64_t i2 = runtimeIndexOfTensor[t2];

    assert(i0 != u64sential && i1 != u64sential && i2 != u64sential);

    uf.unite(static_cast<size_t>(i0), static_cast<size_t>(i1));
    uf.unite(static_cast<size_t>(i0), static_cast<size_t>(i2));
  }

  struct RuntimeGroup {
    memory::vector<uint64_t> tensorIds;
    memory::vector<size_t> constraintIds;
  };

  const memory::vector<memory::vector<size_t>> components = uf.components();
  memory::vector<RuntimeGroup> groups;
  groups.reserve(components.size());

  memory::vector<size_t> groupOfRuntimeIndex(runtimeTensorIds.size(), 0);
  for (size_t i = 0; i < components.size(); ++i) {
    RuntimeGroup g{};
    g.tensorIds.reserve(components[i].size());

    for (size_t j : components[i]) {
      groupOfRuntimeIndex[j] = i;
      const uint64_t tid = runtimeTensorIds[j];
      g.tensorIds.push_back(tid);
    }

    groups.push_back(std::move(g));
  }

  // Attach constraints to their groups.
  for (size_t ci = 0; ci < schedule.memoryConstrains.size(); ++ci) {
    const auto &c = schedule.memoryConstrains[ci];
    const uint64_t t0 = c.src0.index;

    if (isParameter[t0]) {
      diag::not_implemented();
    }

    const size_t rIdx = static_cast<size_t>(runtimeIndexOfTensor[t0]);
    const size_t gi = groupOfRuntimeIndex[rIdx];
    groups[gi].constraintIds.push_back(ci);
  }

  //
  // 4) Materialize runtime buffers + views
  //
  // NOTE: This intentionally supports only the trivial constraint case:
  //   - Either no constraints (group must be size 1),
  //   - Or exactly one implicit concat constraint and group size must be 3.
  // More complex overlap/chain cases should be handled once you decide the
  // intended semantics.

  memory::dynamic_bitset placed(tensorCount, false);
  for (uint64_t tid : paramTensorIds) {
    placed[tid] = true;
  }

  auto create_view = [&](uint64_t tid, uint64_t bufferId, const Sym &offset) {
    if (placed[tid]) {
      diag::unreachable();
    }

    const Tensor &t = schedule.tensors[tid];
    const uint64_t viewId = out.tensors.size();

    out.tensors.push_back(TensorView{
        .buffer = bufferId,
        .offset = offset,
        .size = t.size,
        .info = t.info,
    });

    tensorToView[tid] = viewId;
    placed[tid] = true;
  };

  for (const RuntimeGroup &g : groups) {
    if (g.constraintIds.empty()) {
      // No constraints => only the trivial case is supported: one tensor per
      // group.
      if (g.tensorIds.size() != 1) {
        diag::not_implemented();
      }

      const uint64_t tid = g.tensorIds[0];
      const Tensor &t = schedule.tensors[tid];

      const uint64_t bufferId = out.buffers.size();
      out.buffers.push_back(Buffer{
          .size = t.size,
          .alignment = static_cast<std::uint16_t>(t.alignment),
          .first_use =
              std::numeric_limits<std::uint32_t>::max(), // filled later
          .last_use = 0,
      });

      create_view(tid, bufferId, Sym::Const(0));
      continue;
    }

    // Constrained group: currently only one implicit concat constraint
    // supported.
    if (g.constraintIds.size() != 1) {
      diag::not_implemented();
    }
    if (g.tensorIds.size() != 3) {
      diag::not_implemented();
    }

    const auto &c = schedule.memoryConstrains[g.constraintIds[0]];
    const uint64_t t0 = c.src0.index;
    const uint64_t t1 = c.src1.index;
    const uint64_t t2 = c.dst.index;

    // Require the group to be exactly {t0,t1,t2}.
    {
      auto contains = [&](uint64_t tid) {
        return std::ranges::find(g.tensorIds, tid) != g.tensorIds.end();
      };
      if (!contains(t0) || !contains(t1) || !contains(t2)) {
        diag::unreachable();
      }
    }

    const Tensor &A = schedule.tensors[t0];
    const Tensor &B = schedule.tensors[t1];
    const Tensor &D = schedule.tensors[t2];

    // Buffer size must cover both the concatenation span and dst span.
    // This is safe and uses only add + max (no comparisons needed).
    const Sym concatSpan = sym.add(A.size, B.size);
    const Sym bufferSize = sym.max(D.size, concatSpan);

    std::uint16_t alignment = static_cast<std::uint16_t>(A.alignment);
    alignment = std::max(alignment, static_cast<std::uint16_t>(B.alignment));
    alignment = std::max(alignment, static_cast<std::uint16_t>(D.alignment));

    const uint64_t bufferId = out.buffers.size();
    out.buffers.push_back(Buffer{
        .size = bufferSize,
        .alignment = alignment,
        .first_use = std::numeric_limits<std::uint32_t>::max(), // filled later
        .last_use = 0,
    });

    create_view(t0, bufferId, Sym::Const(0));
    create_view(t1, bufferId, A.size);
    create_view(t2, bufferId, Sym::Const(0));
  }

  // Sanity: all tensors must have a view.
  for (uint64_t tid = 0; tid < tensorCount; ++tid) {
    assert(tensorToView[tid] != u64sential);
  }

  // Remap dispatch bindings: tensor index -> view index

  for (auto &d : out.dispatches) {
    if constexpr (requires { d.bindings; }) {
      for (auto &b : d.bindings) {
        const uint64_t tid = static_cast<uint64_t>(b.tensorId.index);
        assert(tid < tensorCount);
        const uint64_t viewId = tensorToView[tid];
        b.tensorId.index = viewId;
      }
    }
  }

  // Compute buffer lifetimes from dispatch usage

  for (auto &buf : out.buffers) {
    buf.first_use = std::numeric_limits<std::uint32_t>::max();
    buf.last_use = 0;
  }

  for (std::uint32_t di = 0;
       di < static_cast<std::uint32_t>(out.dispatches.size()); ++di) {
    auto &d = out.dispatches[di];

    if constexpr (requires { d.bindings; }) {
      for (const auto &b : d.bindings) {
        const uint64_t viewId = static_cast<uint64_t>(b.tensorId.index);
        assert(viewId < out.tensors.size());

        const uint64_t bufferId = out.tensors[viewId].buffer;
        assert(bufferId < out.buffers.size());

        Buffer &buf = out.buffers[bufferId];
        buf.first_use = std::min(buf.first_use, di);
        buf.last_use = std::max(buf.last_use, di);
      }
    }
  }

  // Normalize unused buffers (if any)
  for (auto &buf : out.buffers) {
    if (buf.first_use == std::numeric_limits<std::uint32_t>::max()) {
      buf.first_use = 0;
      buf.last_use = 0;
    }
  }

  // Remap inputs/outputs (logical tensor ids -> view ids)

  out.inputs.clear();
  out.outputs.clear();

  for (uint64_t tid : schedule.inputs) {
    assert(tid < tensorCount);
    out.inputs.push_back(tensorToView[tid]);
  }

  for (uint64_t tid : schedule.outputs) {
    assert(tid < tensorCount);
    out.outputs.push_back(tensorToView[tid]);
  }

  return out;
}

} // namespace denox::compiler
