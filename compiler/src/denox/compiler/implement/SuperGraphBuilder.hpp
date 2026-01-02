#pragma once

#include "denox/compiler/dce/ConstModel.hpp"
#include "denox/compiler/implement/ComputeDispatchBuilder.hpp"
#include "denox/compiler/implement/OpImpl.hpp"
#include "denox/compiler/implement/SuperGraphEdge.hpp"
#include "denox/compiler/implement/Supergraph.hpp"
#include "denox/compiler/implement/Tensor.hpp"
#include "denox/compiler/specialization/TensorInstance.hpp"

namespace denox::compiler {

class SuperGraphBuilder {
public:
  friend class OpImpl;
  friend class ComputeDispatchBuilder;

  SuperGraphBuilder(const ConstModel &model, SymGraph symGraph)
      : m_symGraph(symGraph) {
    const size_t nodeCount = model.graph.nodeCount();
    for (uint32_t n = 0; n < nodeCount; ++n) {
      memory::NodeId nid{n};
      TensorId tid = createTensor(model.graph.get(nid), nid);
      memory::NodeId _nid = m_graph.addNode(tid);
      assert(_nid == nid);
    }

    m_inputs.assign(model.inputs.begin(), model.inputs.end());
    m_outputs.assign(model.outputs.begin(), model.outputs.end());
  }

  OpImpl beginOp(memory::span<memory::NodeId> inputs, memory::NodeId output) {
    return OpImpl{this, inputs, output};
  }

  SuperGraph finish() {

    return SuperGraph{
        .graph =
            memory::ConstGraph<TensorId, SuperGraphEdge>(std::move(m_graph)),
        .tensors = std::move(m_tensors),
        .inputs = m_inputs,
        .outputs = m_outputs,
        .symGraph = std::move(m_symGraph),
    };
    m_inputs.clear();
    m_outputs.clear();
  }

  SymGraph &symGraph() { return m_symGraph; }

private:
  TensorId createTensor(Sym size, uint16_t alignment, memory::NodeId nodeId) {
    uint64_t nid = *nodeId;
    if (nid >= m_nodeTensorMapping.size()) {
      m_nodeTensorMapping.resize(nid * 2 + 1);
    }
    if (m_nodeTensorMapping[nid].has_value()) {
      uint64_t index = m_nodeTensorMapping[nid]->index;
      assert(m_tensors[index].size == size);
      assert(m_tensors[index].alignment == alignment);
      return TensorId{index};
    }
    TensorId id = createTensor(size, alignment);
    m_nodeTensorMapping[nid] = id;
    return id;
  }

  TensorId createTensor(const TensorInstance &instance, memory::NodeId nodeId) {
    Sym spatialSize = m_symGraph.mul(instance.width, instance.height);
    Sym byteSize = m_symGraph.mul(
        spatialSize, m_symGraph.mul(instance.channels, size_of(instance.type)));
    TensorId id = createTensor(byteSize, align_of(instance.type), nodeId);
    m_tensors[id.index].info = TensorInfo {
      .width = instance.width,
      .height = instance.height,
      .channels = instance.channels,
      .storage = instance.storage,
      .format = instance.format,
      .type = instance.type
    };

    return id;
  }

  TensorId createTensor(Sym size, uint16_t alignment) {
    Tensor tensor{
        .size = size,
        .alignment = alignment,
        .info = {},
    };
    uint64_t index = m_tensors.size();
    m_tensors.emplace_back(std::move(tensor));
    return TensorId{index};
  }

private:
  memory::vector<memory::optional<TensorId>> m_nodeTensorMapping;
  memory::AdjGraph<TensorId, SuperGraphEdge> m_graph;
  memory::vector<Tensor> m_tensors;

  memory::small_vector<memory::NodeId, 2> m_inputs;
  memory::small_vector<memory::NodeId, 2> m_outputs;
  SymGraph m_symGraph;
  bool m_writeParameters = true;
};

} // namespace denox::compiler
