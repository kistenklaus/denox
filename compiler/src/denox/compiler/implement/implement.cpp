#include "denox/compiler/implement/implement.hpp"
#include "denox/algorithm/pattern_matching/match.hpp"
#include "denox/compiler/implement/SuperGraphBuilder.hpp"
#include "denox/compiler/implement/shaders/shaders.hpp"
#include "denox/glsl/GlslCompiler.hpp"
#include "denox/memory/hypergraph/NodeId.hpp"
#include <chrono>
#include <fmt/format.h>

namespace denox::compiler {

SuperGraph implement(const ConstModel &model, const SymGraph &symGraphRef,
                     spirv::GlslCompiler *glslCompiler,
                     const CompileOptions &options, diag::Logger &logger) {

  const size_t nodeCount = model.graph.nodeCount();
  SuperGraphBuilder supergraphBuilder(model, symGraphRef,
                                      options.descriptorPolicies);

  const auto shaders = shaders::get_all_shaders(glslCompiler, options);


  for (size_t s = 0; s < shaders.size(); ++s) {
    const auto &shader = shaders[s];

      const uint32_t percentage = static_cast<uint32_t>(
          std::floor(static_cast<float>((s + 1)) * 50.0f /
                     static_cast<float>(shaders.size() + 1)));

      logger.info(
          "[{:>3}%] {}Generating {} GLSL compute shader configurations{}",
          percentage, logger.green(), shader->name(), logger.reset());

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
