#include "compiler/impl.hpp"
#include "algorithm/pattern_matching/match.hpp"
#include "algorithm/shortest_dag_hyperpath.hpp"
#include "diag/logging.hpp"
#include "heuristic/IHeuristic.hpp"
#include "heuristic/MemoryHeuristic.hpp"
#include "memory/container/dynamic_bitset.hpp"
#include "memory/container/vector.hpp"
#include "memory/hypergraph/AdjGraph.hpp"
#include "memory/hypergraph/ConstGraph.hpp"
#include "shaders/IShader.hpp"
#include "shaders/conv/DirectConvShader.hpp"
#include "shaders/copy/CopyTransformShader.hpp"
#include "shaders/pad/MemoryPadShader.hpp"
#include "shaders/pool/BasicPoolShader.hpp"
#include "shaders/slice/MemorySliceShader.hpp"
#include "shaders/upsample/BasicUpsampleShader.hpp"
#include <exception>

namespace denox::compiler {

struct ComputeOpImpl {
  const IShader *shader;
  unsigned int pattern;
  algorithm::ConstGraphMatch<TensorInstance, ComputeOp> match;
};

using SuperGraph = memory::AdjGraph<TensorInstance, ComputeOpImpl, float>;

void implement(const OpModel &model, const SymGraph &symGraph) {
  const auto &opGraph = model.graph;
  SuperGraph supergraph{};

  for (std::uint64_t n = 0; n < opGraph.nodeCount(); ++n) {
    memory::NodeId nid{n};
    supergraph.addNode(opGraph.get(nid));
  }

  shaders::DirectConvShader directConv;
  shaders::BasicPoolShader basicPool;
  shaders::BasicUpsampleShader basicUpsample;
  shaders::MemoryPadShader memoryPad;
  shaders::MemorySliceShader memorySlice;
  shaders::CopyTransformShader
      copyTransform; // <- TODO matching this one correctly is a pain!

  const IShader *shaders[]{
      &directConv,    //
      &basicPool,     //
      &basicUpsample, //
      &memoryPad,     //
      &memorySlice,   //
      &copyTransform,
  };

  MemoryHeuristic memoryHeuristic{shaders, &opGraph, symGraph, model.input};

  IHeuristic *heuristic = &memoryHeuristic;

  std::size_t nodeCount = opGraph.nodeCount();

  constexpr std::size_t sn = sizeof(shaders) / sizeof(IShader *);
  for (std::size_t s = 0; s < sn; ++s) {

    const IShader *shader = shaders[s];
    const ShaderCapabilities &caps = shader->capabilities();
    const unsigned int pn = static_cast<unsigned int>(caps.patterns.size());
    for (unsigned int p = 0; p < pn; ++p) {
      memory::dynamic_bitset edgeExits(nodeCount * nodeCount * nodeCount,
                                       false);
      for (const auto &m :
           algorithm::match_all(caps.patterns[p].pattern, opGraph)) {
        memory::small_vector<memory::NodeId, 2> inputs;
        memory::small_vector<const TensorInstance *, 2> ins;
        for (std::size_t i = 0; i < caps.patterns[p].inputs.size(); ++i) {
          memory::NodeId in = m[caps.patterns[p].inputs[i]];

          inputs.push_back(in);
          ins.push_back(&opGraph.get(in));
        }
        memory::NodeId out = m[caps.patterns[p].output];
        std::uint64_t edgeId =
            static_cast<std::uint64_t>(inputs[0]) * nodeCount +
            static_cast<std::uint64_t>(out);
        if (inputs.size() == 2) {
          edgeId += inputs[1] * nodeCount * nodeCount;
        }
        if (edgeExits[edgeId]) {
          continue;
        }
        auto pattern = shader->acceptMatch(opGraph, p, m);
        if (!pattern.has_value()) {
          continue;
        }

        edgeExits[edgeId] = true;

        const float w =
            heuristic->eval(ins, opGraph.get(out), *pattern, m, shader);

        supergraph.addEdge(inputs, out,
                           ComputeOpImpl{
                               .shader = shader,
                               .pattern = *pattern,
                               .match = m,
                           },
                           w);
      }
    }
  }
  fmt::println("supergraph edges : {}", supergraph.edgeCount());

  memory::ConstGraph<TensorInstance, ComputeOpImpl, float> constSupergraph{
      supergraph};

  memory::span<const memory::NodeId> starts{&model.input, 1};
  memory::span<const memory::NodeId> ends{&model.output, 1};
  auto hyperpath =
      algorithm::shortest_dag_hyperpath<TensorInstance, ComputeOpImpl, float>(
          constSupergraph, starts, ends);
  if (!hyperpath.has_value()) {
    DENOX_ERROR("Failed to implement model.");
    std::terminate(); // <- TODO proper error handling please
  }

  for (std::size_t op = 0; op < hyperpath->size(); ++op) {
    memory::EdgeId oid{(*hyperpath)[op]};
    const auto& o = constSupergraph.get(oid);
    const auto &srcs = constSupergraph.src(oid);
    bool first = true;
    std::string inStr;
    for (const memory::NodeId &srcId : srcs) {
      const TensorInstance &src = constSupergraph.get(srcId);
      if (!first) {
        inStr += ",";
      }
      first = false;
      inStr += fmt::format("{}[{}]", src.layout.to_string(), src.channels);
    }

    const ComputeOpImpl &impl = constSupergraph.get(oid);
    memory::NodeId dstId = constSupergraph.dst(oid);
    const TensorInstance &dst = constSupergraph.get(dstId);
    fmt::println("{:^22}{:-^40}> {}[{}]", inStr, impl.shader->name(o.pattern),
                 dst.layout.to_string(), dst.channels);
  }

  fmt::println("implementation size : {}", hyperpath->size());
}

} // namespace denox::compiler
