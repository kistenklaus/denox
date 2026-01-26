#include "denox/compiler/selection/selection.hpp"
#include "denox/algorithm/align_up.hpp"
#include "denox/algorithm/minimum_const_subgraph.hpp"
#include "denox/algorithm/prune_dominated_edges.hpp"
#include "denox/algorithm/shortest_dag_hyperpath.hpp"
#include "denox/algorithm/topological_edge_sort.hpp"
#include "denox/algorithm/topological_sort.hpp"
#include "denox/common/TensorFormat.hpp"
#include "denox/compiler/assumed_symeval/assumed_symeval.hpp"
#include "denox/compiler/implement/MemoryConstrain.hpp"
#include "denox/diag/logging.hpp"
#include "denox/diag/not_implemented.hpp"
#include "denox/memory/container/small_vector.hpp"
#include "denox/memory/hypergraph/AdjGraph.hpp"
#include "denox/symbolic/SymGraphEval.hpp"
#include <absl/strings/str_format.h>
#include <fmt/base.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <limits>
#include <stdexcept>

namespace denox::compiler {

using weight_type = std::chrono::duration<float, std::milli>;
static constexpr weight_type INF_WEIGHT = weight_type::max();

OptSchedule select_schedule(SuperGraph &&supergraph, const Db &db,
                            const Model &model, const CompileOptions &options,
                            diag::Logger &logger) {

  logger.info("[ 50%] {}{}Selecting compute shader dispatches{}", logger.bold(),
              logger.green(), logger.reset());

  SymGraphEval eval =
      assumed_symeval(supergraph.symGraph, model.valueNames(), options);

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
        opLatency = INF_WEIGHT;
      }
    }

    weightedSupergraph.addEdge(supergraph.graph.src(eid),
                               supergraph.graph.dst(eid), std::move(edge),
                               opLatency);
  }

  memory::ConstGraph<TensorId, SuperGraphEdge, weight_type>
      constWeightedSupergraph{std::move(weightedSupergraph)};

  // prune duplicate edges
  memory::AdjGraph<TensorId, SuperGraphEdge, weight_type> prunedSupergraph =
      algorithm::prune_duplicate_edges(constWeightedSupergraph);

  memory::ConstGraph<TensorId, SuperGraphEdge, weight_type>
      constPrunedSupergraph{std::move(prunedSupergraph)};

  memory::AdjGraph<TensorId, SuperGraphEdge, weight_type> minimumCostSubgraph =
      algorithm::minimum_cost_subgraph(constWeightedSupergraph,
                                        supergraph.inputs, supergraph.outputs);
  memory::ConstGraph<TensorId, SuperGraphEdge, weight_type> constMinCostGraph(std::move(minimumCostSubgraph));

  memory::vector<memory::EdgeId> minSchedule = algorithm::topological_sort_edges(constMinCostGraph);
  
  // simple union find.
  memory::vector<uint64_t> tensorUf(supergraph.tensors.size());
  for (size_t i = 0; i < tensorUf.size(); ++i) {
    tensorUf[i] = i;
  }

  memory::vector<ComputeDispatch> dispatches;
  memory::vector<MemoryImplicitConcatConstrain> memoryConstrains;
  memory::vector<Parameter> parameters;
  weight_type totalWeight = {};
  for (const auto &eid : minSchedule) {
    totalWeight += constMinCostGraph.weight(eid);
    auto &&edge = constMinCostGraph.get_rvalue(eid);
    if (edge.dispatches.empty() && edge.memoryConstrains.empty() &&
        edge.parameters.empty() && supergraph.graph.src(eid).size() == 1) {

      auto srctid =
          constMinCostGraph.get(constMinCostGraph.src(eid).front()).index;
      auto dsttid = constMinCostGraph.get(constMinCostGraph.dst(eid)).index;
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
  if (totalWeight >= INF_WEIGHT) {
    logger.warn(fmt::format("{}WARNING: Model contains unbenchmarked "
                            "dispatches. Selected schedule is "
                            "most likely suboptimal!{}",
                            logger.yellow(), logger.reset()));
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
  for (auto &c : memoryConstrains) {
    c.dst.index = *tensorRemap[tensorUf[c.dst.index]];
    c.src0.index = *tensorRemap[tensorUf[c.src0.index]];
    c.src1.index = *tensorRemap[tensorUf[c.src1.index]];
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
