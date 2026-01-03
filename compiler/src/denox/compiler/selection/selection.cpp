#include "denox/compiler/selection/selection.hpp"
#include "denox/algorithm/align_up.hpp"
#include "denox/algorithm/shortest_dag_hyperpath.hpp"
#include "denox/common/TensorFormat.hpp"
#include "denox/compiler/implement/MemoryConstrain.hpp"
#include "denox/diag/not_implemented.hpp"
#include "denox/memory/container/small_vector.hpp"
#include "denox/memory/hypergraph/AdjGraph.hpp"
#include "denox/symbolic/SymGraphEval.hpp"
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <limits>
#include <stdexcept>

namespace denox::compiler {

using weight_type = std::chrono::duration<float, std::milli>;

OptSchedule select_schedule(SuperGraph &&supergraph, const Db &db,
                            const Model &model, const Options &options) {

  fmt::println("[ 50%] \x1b[1m\x1b[32mSelecting optimal schedule of compute "
               "dispatches\x1b[0m");

  // eval symgraph!
  memory::small_vector<SymSpec, 4> symSpecs;
  for (const auto &assumption : options.assumptions.valueAssumptions) {
    auto it = std::ranges::find_if(
        model.valueNames(), [&](const NamedValue &namedValue) {
          return namedValue.name == assumption.valueName;
        });
    if (it == model.valueNames().end()) {
      continue;
    }
    auto namedValue = *it;
    if (namedValue.value.isConstant()) {
      continue;
    }
    Sym::symbol symbol = namedValue.value.sym();
    int64_t value = static_cast<int64_t>(assumption.value);
    symSpecs.push_back(SymSpec{.symbol = symbol, .value = value});
  }
  SymGraphEval eval = supergraph.symGraph.eval(symSpecs);

  memory::AdjGraph<TensorId, SuperGraphEdge, weight_type> weightedSupergraph;
  // 1. Add all nodes
  // Because of construction order we know that all node ids of
  // the new weighted and the supergraph match exactly!
  for (size_t n = 0; n < supergraph.graph.nodeCount(); ++n) {
    memory::NodeId nid{n};
    auto tid = supergraph.graph.get(nid);
    memory::NodeId _nid = weightedSupergraph.addNode(tid);
    assert(_nid == nid);
  }
  for (size_t e = 0; e < supergraph.graph.edgeCount(); ++e) {
    memory::EdgeId eid{e};
    auto &&edge = supergraph.graph.get_rvalue(eid);

    weight_type opLatency{};
    for (const auto &dispatch : edge.dispatches) {
      SHA256 hash = dispatch.glsl.fast_sha256();

      memory::small_vector<uint8_t, 4 * 10> pcbuf;
      for (const auto &pc : dispatch.pushConstants) {
        int64_t value;
        if (pc.isDynamic()) {
          Sym::symbol symbol = pc.dynamic();
          value = *eval[Sym::Symbol(symbol)];
        } else {
          switch (pc.type().kind()) {
          case memory::DtypeKind::F16:
          case memory::DtypeKind::F32:
          case memory::DtypeKind::F64:
            diag::not_implemented();
          case memory::DtypeKind::U32:
            value = pc.u32();
            break;
          case memory::DtypeKind::I32:
            value = pc.i32();
            break;
          case memory::DtypeKind::U64:
          case memory::DtypeKind::I64:
            diag::not_implemented();
            break;
          }
        }
        switch (pc.type().kind()) {
        case memory::DtypeKind::F16:
        case memory::DtypeKind::F32:
        case memory::DtypeKind::F64:
          diag::not_implemented();
        case memory::DtypeKind::U32: {
          size_t offset = pcbuf.size();
          offset = algorithm::align_up(offset, alignof(uint32_t));
          uint32_t x = static_cast<uint32_t>(value);
          pcbuf.resize(offset + sizeof(uint32_t), 0);
          std::memcpy(pcbuf.data() + offset, &x, sizeof(uint32_t));

          break;
        }
        case memory::DtypeKind::I32: {
          size_t offset = pcbuf.size();
          offset = algorithm::align_up(offset, alignof(int32_t));
          int32_t x = static_cast<int32_t>(value);
          pcbuf.resize(offset + sizeof(int32_t), 0);
          std::memcpy(pcbuf.data() + offset, &x, sizeof(int32_t));
          break;
        }
        case memory::DtypeKind::U64:
        case memory::DtypeKind::I64:
          diag::not_implemented();
          break;
        }
      }

      uint32_t workgroupCountX =
          static_cast<uint32_t>(*eval[dispatch.workgroupCountX]);
      uint32_t workgroupCountY =
          static_cast<uint32_t>(*eval[dispatch.workgroupCountY]);
      uint32_t workgroupCountZ =
          static_cast<uint32_t>(*eval[dispatch.workgroupCountZ]);

      auto query = db.query_dispatch_latency(hash, pcbuf, workgroupCountX,
                                             workgroupCountY, workgroupCountZ);
      if (query) {
        opLatency += *query;
      } else {
        opLatency = std::numeric_limits<weight_type>::infinity();
      }
    }

    weightedSupergraph.addEdge(supergraph.graph.src(eid),
                               supergraph.graph.dst(eid), std::move(edge),
                               opLatency);
  }

  memory::ConstGraph<TensorId, SuperGraphEdge, weight_type>
      constWeightedSupergraph{std::move(weightedSupergraph)};

  auto path = algorithm::shortest_dag_hyperpath(
      constWeightedSupergraph, supergraph.inputs, supergraph.outputs);

  if (!path) {
    throw std::runtime_error("Failed to implement model!");
  }

  // simple union find.
  memory::vector<uint64_t> tensorUf(supergraph.tensors.size());
  for (size_t i = 0; i < tensorUf.size(); ++i) {
    tensorUf[i] = i;
  }

  memory::vector<ComputeDispatch> dispatches;
  memory::vector<MemoryImplicitConcatConstrain> memoryConstrains;
  memory::vector<Parameter> parameters;
  for (const auto &eid : *path) {
    auto &&edge = constWeightedSupergraph.get_rvalue(eid);
    if (edge.dispatches.empty() && edge.memoryConstrains.empty() &&
        edge.parameters.empty() && supergraph.graph.src(eid).size() == 1) {

      auto srctid =
          supergraph.graph.get(supergraph.graph.src(eid).front()).index;
      auto dsttid = supergraph.graph.get(supergraph.graph.dst(eid)).index;
      const auto &src = supergraph.tensors[srctid];
      const auto &dst = supergraph.tensors[dsttid];
      bool srcIsOpt = src.info.storage == TensorStorage::Optimal ||
                      src.info.format == TensorFormat::Optimal;
      bool dstIsOpt = dst.info.storage == TensorStorage::Optimal ||
                      dst.info.format == TensorFormat::Optimal;

      if (srcIsOpt && dstIsOpt) {
        diag::not_implemented();
      }
      if (srcIsOpt && !dstIsOpt) {
        tensorUf[srctid] = dsttid;
      }
      if (!srcIsOpt && dstIsOpt) {
        tensorUf[dsttid] = srctid;
      }

      continue;
    }
    for (auto &&d : std::move(edge.dispatches)) {
      dispatches.emplace_back(std::move(d));
    }
    for (auto &&c : std::move(edge.memoryConstrains)) {
      memoryConstrains.emplace_back(std::move(c));
    }
    for (auto &&p : std::move(edge.parameters)) {
      parameters.emplace_back(std::move(p));
    }
  }

  memory::vector<memory::optional<uint64_t>> tensorRemap(
      supergraph.tensors.size(), memory::nullopt);

  memory::vector<Tensor> tensors;

  for (auto &d : dispatches) {
    for (auto &b : d.bindings) {
      uint64_t tid = tensorUf[b.tensorId.index];
      if (tensorRemap[tid].has_value()) {
        b.tensorId.index = *tensorRemap[tid];
        continue;
      }
      const uint64_t new_tid = tensors.size();
      b.tensorId.index = new_tid;
      tensorRemap[tid] = new_tid;
      tensors.emplace_back(std::move(supergraph.tensors[tid]));
    }
  }

  // rebind parameters.
  for (auto &p : parameters) {
    assert(tensorRemap[tensorUf[p.tensorId.index]].has_value());
    p.tensorId.index = *tensorRemap[tensorUf[p.tensorId.index]];
  }
  // rebind inputs
  memory::vector<uint64_t> inputs;
  for (const auto &nid : supergraph.inputs) {
    auto tid = tensorUf[supergraph.graph.get(nid).index];
    assert(tensorRemap[tid].has_value());
    inputs.push_back(*tensorRemap[tid]);
  }
  // rebind outputs
  memory::vector<uint64_t> outputs;
  for (const auto &nid : supergraph.outputs) {
    auto tid = tensorUf[supergraph.graph.get(nid).index];
    assert(tensorRemap[tid].has_value());
    outputs.push_back(*tensorRemap[tid]);
  }

  return OptSchedule{
      .symGraph = std::move(supergraph.symGraph),
      .tensors = std::move(tensors),
      .dispatches = std::move(dispatches),
      .memoryConstrains = std::move(memoryConstrains),
      .parameters = std::move(parameters),
      .inputs = std::move(inputs),
      .outputs = std::move(outputs),
  };

  // auto path = algorithm::shortest_dag_hyperpath(supergraph.graph,
  // supergraph.inputs, supergraph.outputs);
}

} // namespace denox::compiler
