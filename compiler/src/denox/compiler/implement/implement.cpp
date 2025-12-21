#include "denox/compiler/implement/implement.hpp"
#include "denox/compiler/implement/Supergraph.hpp"
#include "denox/compiler/implement/shaders/shaders.hpp"
#include "denox/compiler/specialization/TensorInstance.hpp"
#include "denox/glsl/GlslCompiler.hpp"
#include "denox/memory/hypergraph/AdjGraph.hpp"
#include "denox/spirv/ShaderDebugInfoLevel.hpp"
#include "denox/spirv/SpirvTools.hpp"

namespace denox::compiler {

SuperGraph implement(const ConstModel &model, const SymGraph &symGraphRef,
                     const Options &options) {
  SymGraph symGraph = symGraphRef;
  memory::AdjGraph<TensorInstance, SuperGraphEdge> adjGraph;

  for (uint32_t n = 0; n < model.graph.nodeCount(); ++n) {
    memory::NodeId nid{n};
    memory::NodeId _nid = adjGraph.addNode(model.graph.get(nid));
    assert(_nid == nid);
  }

  spirv::SpirvTools spvTools{options.deviceInfo};
  spirv::GlslCompiler glslCompiler{&spvTools, options.deviceInfo};

  const auto shaders = shaders::get_all_shaders(&glslCompiler, options);

  return {
      .graph = memory::ConstGraph<TensorInstance, SuperGraphEdge>{adjGraph},
      .symGraph = std::move(symGraph),
  };
}

} // namespace denox::compiler
