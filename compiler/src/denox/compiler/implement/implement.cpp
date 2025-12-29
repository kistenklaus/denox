#include "denox/compiler/implement/implement.hpp"
#include "denox/algorithm/pattern_matching/match.hpp"
#include "denox/compiler/implement/SuperGraphBuilder.hpp"
#include "denox/compiler/implement/shaders/shaders.hpp"
#include "denox/glsl/GlslCompiler.hpp"
#include "denox/memory/hypergraph/NodeId.hpp"
#include "denox/spirv/SpirvTools.hpp"
#include <chrono>
#include <exception>
#include <ratio>

namespace denox::compiler {

SuperGraph implement(const ConstModel &model, const SymGraph &symGraphRef,
                     const Options &options) {

  const size_t nodeCount = model.graph.nodeCount();
  SuperGraphBuilder supergraphBuilder(model, symGraphRef);

  spirv::SpirvTools spvTools{options.deviceInfo};
  spirv::GlslCompiler glslCompiler{&spvTools, options.deviceInfo};

  const auto shaders = shaders::get_all_shaders(&glslCompiler, options);

  float sum = 0;
  for (const auto &shader : shaders) {
    auto start = std::chrono::high_resolution_clock::now();

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

          auto s2 = std::chrono::high_resolution_clock::now();

          opImpl.finish();

          auto dur = std::chrono::high_resolution_clock::now() - s2;
          auto durms = std::chrono::duration_cast<
              std::chrono::duration<float, std::milli>>(dur);
          // fmt::println("impl-took {}ms", durms.count());
        }
      }
    }
    auto dur = std::chrono::high_resolution_clock::now() - start;
    auto durms =
        std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(
            dur);
    fmt::println("{} took {}ms", shader->name(0, 0), durms.count());
  }
  // fmt::println("total: {}ms", sum);

  return supergraphBuilder.finish();
}

} // namespace denox::compiler
