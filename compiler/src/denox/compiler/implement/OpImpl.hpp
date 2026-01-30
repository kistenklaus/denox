#pragma once

#include "denox/common/TensorDataType.hpp"
#include "denox/common/TensorFormat.hpp"
#include "denox/common/TensorStorage.hpp"
#include "denox/compiler/implement/ComputeDispatch.hpp"
#include "denox/compiler/implement/ComputeDispatchBuilder.hpp"
#include "denox/compiler/implement/MemoryConstrain.hpp"
#include "denox/compiler/implement/Parameter.hpp"
#include "denox/compiler/implement/SuperGraphEdge.hpp"
#include "denox/compiler/implement/TensorId.hpp"
#include "denox/glsl/GlslCompilerInstance.hpp"
#include "denox/memory/container/optional.hpp"
#include "denox/memory/container/small_vector.hpp"
#include "denox/memory/hypergraph/NodeId.hpp"
#include "denox/memory/tensor/BiasDescriptor.hpp"
#include "denox/memory/tensor/BiasTensor.hpp"
#include "denox/memory/tensor/FilterTensor.hpp"
#include "denox/memory/tensor/FitlerDescriptor.hpp"
#include "denox/symbolic/Sym.hpp"
#include "denox/symbolic/SymGraph.hpp"
namespace denox::compiler {

class SuperGraphBuilder;

class OpImpl {
public:
  friend class SuperGraphBuilder;
  friend class ComputeDispatchBuilder;

  TensorId createParameter(size_t elemCount, TensorDataType dtype,
                           TensorStorage storage,
                           TensorFormat format,
                           std::function<std::vector<std::byte>()> value,
                           uint16_t alignment = 16);

  ComputeDispatchBuilder registerDispatch(spirv::GlslCompilerInstance glsl,
                                          Sym wgX, Sym wgY = Sym::Const(1),
                                          Sym wgZ = Sym::Const(1));

  void createImplicitConcatConstrain(memory::NodeId src0, memory::NodeId src1,
                                     memory::NodeId dst);

  void finish();

private:
  OpImpl(SuperGraphBuilder *builder, memory::span<const memory::NodeId> inputs,
         memory::NodeId output)
      : m_superBuilder(builder), m_inputs(inputs.begin(), inputs.end()),
        m_output(output) {}

private:
  SuperGraphBuilder *m_superBuilder;
  memory::small_vector<memory::NodeId, 2> m_inputs;
  memory::NodeId m_output;

  memory::small_vector<ComputeDispatch, SuperGraphEdge::DISPATCH_SVO> m_dispatches;
  memory::vector<MemoryImplicitConcatConstrain> m_memoryConstrains;
  memory::small_vector<Parameter, SuperGraphEdge::PARAM_SVO> m_parameters;
};

} // namespace denox::compiler
