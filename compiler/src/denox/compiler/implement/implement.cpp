#include "denox/compiler/implement/implement.hpp"
#include "denox/algorithm/pattern_matching/match.hpp"
#include "denox/compiler/implement/Supergraph.hpp"
#include "denox/compiler/implement/shaders/shaders.hpp"
#include "denox/compiler/specialization/TensorInstance.hpp"
#include "denox/glsl/GlslCompiler.hpp"
#include "denox/memory/hypergraph/AdjGraph.hpp"
#include "denox/memory/hypergraph/NodeId.hpp"
#include "denox/spirv/SpirvTools.hpp"

namespace denox::compiler {

SuperGraph implement(const ConstModel &model, const SymGraph &symGraphRef,
                     const Options &options) {
  SymGraph symGraph = symGraphRef;
  memory::AdjGraph<TensorInstance, SuperGraphEdge> adjGraph;

  const size_t nodeCount = model.graph.nodeCount();
  for (uint32_t n = 0; n < nodeCount; ++n) {
    memory::NodeId nid{n};
    memory::NodeId _nid = adjGraph.addNode(model.graph.get(nid));
    assert(_nid == nid);
  }

  spirv::SpirvTools spvTools{options.deviceInfo};
  spirv::GlslCompiler glslCompiler{&spvTools, options.deviceInfo};

  const auto shaders = shaders::get_all_shaders(&glslCompiler, options);

  for (const auto &shader : shaders) {
    const ShaderCapabilities &caps = shader->capabilities();
    for (uint32_t p = 0; p < caps.patterns.size(); ++p) {
      const auto &pattern = caps.patterns[p];
      std::unordered_set<uint64_t> edgeExists;

      for (const auto &m : algorithm::match_all(pattern.pattern, model.graph)) {
        memory::NodeId output = m[pattern.output];
        memory::small_vector<memory::NodeId, 2> inputs;
        uint64_t edgeId = 0;
        for (const auto &inputMatch : pattern.inputs) {
          memory::NodeId input = m[inputMatch];
          inputs.push_back(input);
          edgeId = edgeId * nodeCount + *input;
        }
        edgeId = edgeId * nodeCount + *output;
        if (edgeExists.contains(edgeId)) {
          continue;
        }
        edgeExists.insert(edgeId);

        const auto configs = shader->acceptMatch(model.graph, p, m);
        for (const auto &config : configs) {
          // shader->implement(impl, model.graph, p, config, m, symGraph);
          adjGraph.addEdge(inputs, output, SuperGraphEdge{});
        }
      }
    }
  }

  return {
      .graph = memory::ConstGraph<TensorInstance, SuperGraphEdge>{adjGraph},
      .symGraph = std::move(symGraph),
  };
}

} // namespace denox::compiler
