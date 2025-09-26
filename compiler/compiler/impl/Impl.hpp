#pragma once

#include "compiler/impl/ComputeDispatchBuilder.hpp"
#include "compiler/ir/TensorInstance.hpp"
#include "compiler/ir/impl/ImplModel.hpp"
#include "compiler/ir/impl/TensorId.hpp"
#include "compiler/ir/impl/TensorStorageRequirements.hpp"
#include "memory/container/vector.hpp"
#include "memory/hypergraph/NodeId.hpp"
#include "memory/tensor/FilterTensor.hpp"
#include <stdexcept>
namespace denox::compiler {

class Impl {
public:
  friend class ComputeDispatchBuilder;
  Impl(ImplModel *impl) : m_impl(impl) {}

  Impl(const Impl &) = delete;
  Impl(Impl &&) = delete;
  Impl &operator=(const Impl &) = delete;
  Impl &operator=(Impl &&) = delete;

  TensorId createTensor(Sym byteSize, unsigned int minAlignment) {
    TensorStorageRequirements req{
        .byteSize = byteSize,
        .minAlignment = minAlignment,
        .meta = nullptr,
    };
    std::uint64_t index = m_impl->tensors.size();
    m_impl->tensors.emplace_back(std::move(req));
    return TensorId{index};
  }

  /// NOTE: May return already existing tensor.
  TensorId createTensor(Sym byteSize, unsigned int minAlignment,
                        memory::NodeId nodeId) {
    std::uint64_t nid = static_cast<std::uint64_t>(nodeId);
    if (nid >= m_nodeTensorMapping.size()) {
      m_nodeTensorMapping.resize(nid * 2 + 1);
    }
    if (m_nodeTensorMapping[nid].has_value()) {
      assert(m_impl->tensors[m_nodeTensorMapping[nid]->index].byteSize ==
             byteSize);
      assert(m_impl->tensors[m_nodeTensorMapping[nid]->index].minAlignment ==
             minAlignment);

      return m_nodeTensorMapping[nid].value();
    }

    TensorId id = createTensor(byteSize, minAlignment);
    m_nodeTensorMapping[nid] = id;
    return id;
  }

  /// NOTE: May return already existing tensor.
  TensorId createTensor(SymGraph &symGraph, const TensorInstance &instance,
                        memory::NodeId nodeId) {
    Sym spatialSize =
        symGraph.mul(instance.extent.x.asSym(), instance.extent.y.asSym());
    Sym byteSize =
        symGraph.mul(spatialSize, instance.channels * instance.type.size());
    return createTensor(
        byteSize, static_cast<unsigned int>(instance.type.alignment()), nodeId);
  }

  TensorId createParameter(memory::vector<std::byte> data,
                           unsigned int minAlignment) {
    TensorId id = createTensor(Sym::Const(data.size()), minAlignment);
    m_impl->parameters.emplace_back(id, std::move(data));
    return id;
  }

  TensorId createParameter(memory::FilterTensorConstView filterTensor) {
    auto bytes = filterTensor.span();
    unsigned int minAlignment;
    if (filterTensor.layout().isVectorized()) {
      minAlignment = 16;
    } else {
      minAlignment = static_cast<unsigned int>(filterTensor.type().alignment());
    }
    return createParameter({bytes.begin(), bytes.end()}, minAlignment);
  }

  TensorId createParameter(memory::BiasTensorConstView biasTensor) {
    auto bytes = biasTensor.span();
    unsigned int minAlignment;
    if (biasTensor.layout().isVectorized()) {
      minAlignment = 16;
    } else {
      minAlignment = static_cast<unsigned int>(biasTensor.type().alignment());
    }
    return createParameter({bytes.begin(), bytes.end()}, minAlignment);
  }

  ComputeDispatchBuilder dispatch(memory::vector<std::uint32_t> spirvSrc) {
    auto builder = ComputeDispatchBuilder(m_impl->dispatches.size(), this);
    m_impl->dispatches.push_back({});
    m_impl->dispatches.back().spirvSrc = std::move(spirvSrc);
    return builder;
  }

  void createMemoryPlacementConstrain(TensorId src0, TensorId src1,
                                      TensorId dst) {
    m_impl->memoryImplicitConcatConstrains.emplace_back(src0, src1, dst);
  }

  /// Ensures that tensors nodeA and nodeB memory is placed directly after one
  /// another.
  void createImplicitConcatConstrain(memory::NodeId src0, memory::NodeId src1,
                                     memory::NodeId dst) {
    memory::optional<TensorId> tensorSrc0 = tryRemap(src0);
    memory::optional<TensorId> tensorSrc1 = tryRemap(src1);
    memory::optional<TensorId> tensorDst = tryRemap(dst);
    if (tensorSrc0.has_value() && tensorSrc1.has_value() &&
        tensorDst.has_value()) {
      createMemoryPlacementConstrain(*tensorSrc0, *tensorSrc1, *tensorDst);
    } else {
      throw std::runtime_error(
          "Failed to create memory constrain, invalid node ids");
    }
  }

private:
  memory::optional<TensorId> tryRemap(memory::NodeId nodeId) {
    if (nodeId >= m_nodeTensorMapping.size()) {
      return memory::nullopt;
    }
    if (m_nodeTensorMapping[nodeId].has_value()) {
      return m_nodeTensorMapping[nodeId].value();
    } else {
      return memory::nullopt;
    }
  }

  memory::vector<memory::optional<TensorId>> m_nodeTensorMapping;
  ImplModel *m_impl;
};

} // namespace denox::compiler
