#include "compiler/impl.hpp"
#include "algorithm/pattern_matching/match.hpp"
#include "memory/container/dynamic_bitset.hpp"
#include "memory/hypergraph/AdjGraph.hpp"
#include "shaders/IShader.hpp"
#include "shaders/conv/DirectConvShader.hpp"
#include "shaders/copy/CopyTransformShader.hpp"
#include "shaders/pad/MemoryPadShader.hpp"
#include "shaders/pool/BasicPoolShader.hpp"
#include "shaders/slice/MemorySliceShader.hpp"
#include "shaders/upsample/BasicUpsampleShader.hpp"

namespace denox::compiler {

struct ComputeOpImpl {
  const IShader *shader;
  unsigned int pattern;
  algorithm::ConstGraphMatch<ComputeTensor, ComputeOp> match;
};

static float heuristic([[maybe_unused]] std::span<const ComputeTensor *> in,
                       [[maybe_unused]] const ComputeTensor &out,
                       [[maybe_unused]] unsigned int pattern,
                       [[maybe_unused]] unsigned int shader) {
  // Big TODO!!
  return 1.0f;
}

void implement(const ConstModel &model) {
  const auto &opGraph = model.graph;
  using SuperGraph = memory::AdjGraph<ComputeTensor, ComputeOpImpl, float>;
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
  shaders::CopyTransformShader copyTransform;

  IShader *shaders[]{
      &directConv,    //
      &basicPool,     //
      &basicUpsample, //
      &memoryPad,     //
      &memorySlice,   //
      &copyTransform,
  };

  std::size_t nodeCount = opGraph.nodeCount();

  constexpr std::size_t sn = sizeof(shaders) / sizeof(IShader *);
  for (std::size_t s = 0; s < sn; ++s) {
    memory::dynamic_bitset edgeExits(nodeCount * nodeCount * nodeCount);

    const IShader *shader = shaders[s];
    const ShaderCapabilities &caps = shader->capabilities();
    const unsigned int pn = static_cast<unsigned int>(caps.patterns.size());
    for (unsigned int p = 0; p < pn; ++p) {
      for (const auto &m :
           algorithm::match_all(caps.patterns[p].pattern, opGraph)) {
        memory::small_vector<memory::NodeId, 2> inputs;
        memory::small_vector<const ComputeTensor *, 2> ins;
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
        edgeExits[edgeId] = true;

        const float w =
            heuristic(ins, opGraph.get(out), p, static_cast<unsigned int>(s));
        supergraph.addEdge(inputs, out,
                           ComputeOpImpl{
                               .shader = shader,
                               .pattern = p,
                               .match = m,
                           },
                           w);
      }
    }
  }
  fmt::println("supergraph edges : {}", supergraph.edgeCount());
}

} // namespace denox::compiler
