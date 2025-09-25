#include "compiler/impl.hpp"
#include "algorithm/pattern_matching/match.hpp"
#include "memory/container/dynamic_bitset.hpp"
#include "memory/hypergraph/AdjGraph.hpp"
#include "shaders/IShader.hpp"
#include "shaders/conv/DirectConvShader.hpp"
#include "shaders/pad/BasicPadShader.hpp"
#include "shaders/pool/BasicPoolShader.hpp"
#include "shaders/upsample/BasicUpsampleShader.hpp"

namespace denox::compiler {

struct ComputeOpImpl {
  const IShader *shader;
  unsigned int pattern;
  algorithm::ConstGraphMatch<ComputeTensor, ComputeOp> match;
};

static float heuristic([[maybe_unused]] const ComputeTensor &in,
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
  shaders::BasicPadShader basicPad;

  IShader *shaders[]{
      &directConv,
      &basicPool,
      &basicUpsample,
      &basicPad,
  };

  std::size_t nodeCount = opGraph.nodeCount();

  constexpr std::size_t sn = sizeof(shaders) / sizeof(IShader *);
  for (std::size_t s = 0; s < sn; ++s) {
    memory::dynamic_bitset edgeExits(nodeCount * nodeCount);

    const IShader *shader = shaders[s];
    const ShaderCapabilities &caps = shader->capabilities();
    const unsigned int pn = static_cast<unsigned int>(caps.patterns.size());
    for (unsigned int p = 0; p < pn; ++p) {
      for (const auto &m :
           algorithm::match_all(caps.patterns[p].pattern, opGraph)) {
        memory::NodeId in = m[caps.patterns[p].input];
        memory::NodeId out = m[caps.patterns[p].output];
        std::uint64_t edgeId = static_cast<std::uint64_t>(in) * nodeCount +
                               static_cast<std::uint64_t>(out);
        if (edgeExits[edgeId]) {
          continue;
        }
        edgeExits[edgeId] = true;

        const float w = heuristic(opGraph.get(in), opGraph.get(out), p,
                                  static_cast<unsigned int>(s));
        supergraph.addEdge(in, out,
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
