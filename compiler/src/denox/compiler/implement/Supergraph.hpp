#pragma once

#include "denox/compiler/implement/PushConstant.hpp"
#include "denox/compiler/implement/TensorBinding.hpp"
#include "denox/compiler/specialization/TensorInstance.hpp"
#include "denox/glsl/GlslCompilerInstance.hpp"
#include "denox/memory/hypergraph/ConstGraph.hpp"
#include "denox/symbolic/SymGraph.hpp"

namespace denox::compiler {


struct ComputeDispatchInfo {
  memory::optional<memory::string> name;
  memory::optional<memory::string> debug_info;
  memory::optional<io::Path> srcPath;
  memory::optional<Sym> memoryReads;
  memory::optional<Sym> memoryWrites;
};

struct GlslComputeDispatch {
  spirv::GlslCompilerInstance glsl;
  PushConstant pushConstant;
  uint32_t workgroupCountX;
  uint32_t workgroupCountY;
  uint32_t workgroupCountZ;
  memory::vector<TensorBinding> bindings;
};

struct SuperGraphEdge {
  memory::vector<GlslComputeDispatch> dispatches;
};

struct SuperGraph {
  memory::ConstGraph<TensorInstance, SuperGraphEdge> graph;
  SymGraph symGraph;
};

} // namespace denox::compiler
