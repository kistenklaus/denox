#include "denox/compiler/implement/implement.hpp"
#include "denox/algorithm/pattern_matching/match.hpp"
#include "denox/compiler/implement/SuperGraphBuilder.hpp"
#include "denox/compiler/implement/shaders/shaders.hpp"
#include "denox/glsl/GlslCompiler.hpp"
#include "denox/memory/hypergraph/NodeId.hpp"
#include <fmt/format.h>

namespace denox::compiler {

SuperGraph implement(const ConstModel &model, const SymGraph &symGraphRef,
                     spirv::GlslCompiler *glslCompiler,
                     const Options &options) {

  const size_t nodeCount = model.graph.nodeCount();
  SuperGraphBuilder supergraphBuilder(model, symGraphRef);

  const auto shaders = shaders::get_all_shaders(glslCompiler, options);

  size_t totalPatterns = 0;
  for (const auto &shader : shaders) {
    totalPatterns += shader->capabilities().patterns.size();
  }

  size_t pp = 0;

  for (size_t s = 0; s < shaders.size(); ++s) {
    const auto &shader = shaders[s];

    const ShaderCapabilities &caps = shader->capabilities();
    for (uint32_t p = 0; p < caps.patterns.size(); ++p, ++pp) {

      uint32_t percentage = static_cast<uint32_t>(
          std::floor(static_cast<float>((pp + 1)) * 50.0f /
                     static_cast<float>(totalPatterns + 1)));
      fmt::println("[{:>3}%] \x1B[34mGenerating \x1b[1m{}\x1B[0m\x1B[34m GLSL compute shader configurations\x1B[0m", percentage, shader->name(p, 0));

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
        const auto configs = shader->acceptMatch(model.graph, p, m);
        if (configs.empty()) {
          continue;
        }

        edgeExists.insert(edgeId);

        for (const auto &config : configs) {

          auto opImpl = supergraphBuilder.beginOp(inputs, output);
          shader->implement(opImpl, model.graph, p, config, m,
                            supergraphBuilder.symGraph());
          opImpl.finish();
        }
      }
    }
  }
  // fmt::println("total: {}ms", sum);

  return supergraphBuilder.finish();
}

} // namespace denox::compiler
