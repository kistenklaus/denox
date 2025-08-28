#pragma once

#include "vkcnn/common/FilterMode.hpp"
#include "vkcnn/common/PaddingMode.hpp"
#include "vkcnn/common/PoolFunction.hpp"
#include "vkcnn/common/hypergraph/AdjGraph.hpp"
#include "vkcnn/common/hypergraph/ConstGraph.hpp"
#include "vkcnn/common/hypergraph/NodeId.hpp"
#include "vkcnn/common/model/ComputeOp.hpp"
#include "vkcnn/common/model/ComputeTensor.hpp"
#include "vkcnn/common/model/SymTensorExtent.hpp"
#include "vkcnn/common/symbolic/SymGraph.hpp"
#include "vkcnn/common/symbolic/Symbolic.hpp"
#include "vkcnn/common/tensor/ActivationLayout.hpp"
#include "vkcnn/common/tensor/FloatType.hpp"
#include <functional>
#include <glm/fwd.hpp>
#include <glm/vec2.hpp>
#include <memory>
#include <optional>
#include <stdexcept>

namespace vkcnn {

class Model;
class Tensor;

namespace details {

struct ComputeGraphControlBlock {
  friend Model;
  friend Tensor;

  static constexpr hypergraph::NodeId NullNode{static_cast<std::size_t>(-1)};

  ComputeGraphControlBlock()
      : input(NullNode), output(NullNode), hypergraph(),
        symGraph(std::make_shared<vkcnn::SymGraph>()),
        inputExtent(symGraph->var(), symGraph->var()) {}

  vkcnn::hypergraph::NodeId input;
  vkcnn::hypergraph::NodeId output;
  vkcnn::hypergraph::AdjGraph<ComputeTensor, ComputeOp> hypergraph;

  std::shared_ptr<vkcnn::SymGraph> symGraph;
  SymTensorExtent inputExtent;
};

}; // namespace details

class Tensor {
public:
  friend Model;

  std::optional<ActivationLayout> layout() const {
    return m_controlBlock->hypergraph.get(m_nodeId).m_layout;
  }

  void setLayout(std::optional<ActivationLayout> layout) {
    m_controlBlock->hypergraph.get(m_nodeId).m_layout = layout;
  }

  std::optional<FloatType> type() const {
    return m_controlBlock->hypergraph.get(m_nodeId).m_type;
  }

  void setType(std::optional<FloatType> type) {
    m_controlBlock->hypergraph.get(m_nodeId).m_type = type;
  }

  unsigned int channels() const {
    return m_controlBlock->hypergraph.get(m_nodeId).m_channels;
  }

  Symbolic width() const {
    return Symbolic(m_controlBlock->symGraph,
                    m_controlBlock->hypergraph.get(m_nodeId).m_extent.width);
  }

  Symbolic height() const {
    return Symbolic(m_controlBlock->symGraph,
                    m_controlBlock->hypergraph.get(m_nodeId).m_extent.height);
  }

private:
  Tensor(hypergraph::NodeId id,
         std::shared_ptr<details::ComputeGraphControlBlock> controlBlock)
      : m_nodeId(id), m_controlBlock(std::move(controlBlock)) {}

  hypergraph::NodeId m_nodeId;
  std::shared_ptr<details::ComputeGraphControlBlock> m_controlBlock;
};

class Model {
public:
  Model()
      : m_controlBlock(std::make_shared<details::ComputeGraphControlBlock>()) {}

  Tensor input(unsigned int channels,
               std::optional<ActivationLayout> layout = std::nullopt,
               std::optional<FloatType> type = std::nullopt) {
    assert(m_controlBlock->input ==
           details::ComputeGraphControlBlock::NullNode);
    hypergraph::NodeId id = m_controlBlock->hypergraph.emplaceNode(
        m_controlBlock->inputExtent, channels, layout, type);
    m_controlBlock->input = id;
    return Tensor{id, m_controlBlock};
  }

  Tensor conv2d(const Tensor &src, glm::uvec2 kernelSize, unsigned int K,
                bool bias, glm::uvec2 stride, glm::uvec2 padding,
                std::optional<FloatType> atype = std::nullopt) {

    hypergraph::NodeId srcId = src.m_nodeId;
    const auto &srcTensor = m_controlBlock->hypergraph.get(srcId);
    SymTensorExtent extent{
        m_controlBlock->symGraph->pool(srcTensor.m_extent.width, kernelSize.x,
                                       padding.x, stride.x),
        m_controlBlock->symGraph->pool(srcTensor.m_extent.height, kernelSize.y,
                                       padding.y, stride.y)};

    hypergraph::NodeId dstId = m_controlBlock->hypergraph.emplaceNode(
        extent, K, std::nullopt, std::nullopt);

    m_controlBlock->hypergraph.addEdge(
        srcId, dstId,
        ComputeOp{ComputeOpConv(kernelSize, K, bias, padding, stride, atype)});
    return Tensor{dstId, m_controlBlock};
  }

  Tensor activation(const Tensor &src, ActivationFunction func) {
    hypergraph::NodeId srcId = src.m_nodeId;
    const auto &srcNode = m_controlBlock->hypergraph.get(srcId);

    hypergraph::NodeId dstId = m_controlBlock->hypergraph.emplaceNode(
        srcNode.m_extent, srcNode.m_channels, std::nullopt, std::nullopt);

    m_controlBlock->hypergraph.addEdge(srcId, dstId,
                                       ComputeOp{ComputeOpActivation{func}});
    return Tensor{dstId, m_controlBlock};
  }

  Tensor upsample(const Tensor &src, unsigned int scalingFactor,
                  FilterMode mode) {
    hypergraph::NodeId srcId = src.m_nodeId;
    const auto &srcNode = m_controlBlock->hypergraph.get(srcId);

    SymTensorExtent extent{
        m_controlBlock->symGraph->mul(scalingFactor, srcNode.m_extent.width),
        m_controlBlock->symGraph->mul(scalingFactor, srcNode.m_extent.height)};

    hypergraph::NodeId dstId = m_controlBlock->hypergraph.emplaceNode(
        extent, srcNode.m_channels, std::nullopt, std::nullopt);

    m_controlBlock->hypergraph.addEdge(
        srcId, dstId, ComputeOp{ComputeOpUpsample{scalingFactor, mode}});
    return Tensor{dstId, m_controlBlock};
  }

  Tensor pool(const Tensor &src, glm::uvec2 kernelSize, glm::uvec2 padding,
              glm::uvec2 stride, PoolFunction poolFunc) {

    hypergraph::NodeId srcId = src.m_nodeId;
    const auto &srcNode = m_controlBlock->hypergraph.get(srcId);
    SymTensorExtent extent{
        m_controlBlock->symGraph->pool(srcNode.m_extent.width, kernelSize.x,
                                       padding.x, stride.x),
        m_controlBlock->symGraph->pool(srcNode.m_extent.height, kernelSize.y,
                                       padding.y, stride.y)};

    hypergraph::NodeId dstId = m_controlBlock->hypergraph.emplaceNode(
        extent, srcNode.m_channels, std::nullopt, std::nullopt);

    m_controlBlock->hypergraph.addEdge(
        srcId, dstId,
        ComputeOp{ComputeOpPool(kernelSize, padding, stride, poolFunc)});
    return Tensor{dstId, m_controlBlock};
  }

  Tensor concat(const Tensor &src0, const Tensor &src1) {
    hypergraph::NodeId src0Id = src0.m_nodeId;
    auto src0Node = m_controlBlock->hypergraph.get(src0Id);

    hypergraph::NodeId src1Id = src1.m_nodeId;
    auto src1Node = m_controlBlock->hypergraph.get(src1Id);

    if (src0Node.m_extent != src1Node.m_extent) {
      throw std::runtime_error(
          "Concat arguments, must provably have the same image extent.");
    }


    hypergraph::NodeId dstId = m_controlBlock->hypergraph.emplaceNode(
        src0Node.m_extent, src0Node.m_channels + src1Node.m_channels,
        std::nullopt, std::nullopt);

    m_controlBlock->hypergraph.addEdge(src0Id, src1Id, dstId,
                                       ComputeOp{ComputeOpConcat{}});

    return Tensor{dstId, m_controlBlock};
  }

  template <typename L, typename R, typename T, typename B>
    requires(std::same_as<L, Sym> || std::is_integral_v<L>) &&
            (std::same_as<R, Sym> || std::is_integral_v<R>) &&
            (std::same_as<T, Sym> || std::is_integral_v<T>) &&
            (std::same_as<B, Sym> || std::is_integral_v<B>)
  Tensor pad(const Tensor &src0, L left, R right, T top, B bottom,
             PaddingMode mode) {

    auto srcId = src0.m_nodeId;
    auto srcNode = m_controlBlock->hypergraph.get(srcId);

    SymTensorExtent extent{
        m_controlBlock->symGraph->add(
            srcNode.m_extent.width,
            m_controlBlock->symGraph->add(left, right, false)),
        m_controlBlock->symGraph->add(
            srcNode.m_extent.height,
            m_controlBlock->symGraph->add(top, bottom, false)),
    };
    auto dstId = m_controlBlock->hypergraph.emplaceNode(
        extent, srcNode.channels(), std::nullopt, std::nullopt);

    m_controlBlock->hypergraph.addEdge(
        srcId, dstId,
        ComputeOp{ComputeOpPad(m_controlBlock->symGraph->resolve(left),
                               m_controlBlock->symGraph->resolve(right),
                               m_controlBlock->symGraph->resolve(top),
                               m_controlBlock->symGraph->resolve(bottom),
                               mode)});
    return Tensor{dstId, m_controlBlock};
  }

  template <typename L, typename R, typename T, typename B>
    requires(std::same_as<L, Sym> || std::is_integral_v<L>) &&
            (std::same_as<R, Sym> || std::is_integral_v<R>) &&
            (std::same_as<T, Sym> || std::is_integral_v<T>) &&
            (std::same_as<B, Sym> || std::is_integral_v<B>)
  Tensor slice(const Tensor &src0, L left, R right, T top, B bottom) {

    auto srcId = src0.m_nodeId;
    auto srcNode = m_controlBlock->hypergraph.get(srcId);

    SymTensorExtent extent{
        m_controlBlock->symGraph->sub(right, left),
        m_controlBlock->symGraph->add(bottom, top),
    };
    auto dstId = m_controlBlock->hypergraph.emplaceNode(
        extent, srcNode.channels(), std::nullopt, std::nullopt);

    m_controlBlock->hypergraph.addEdge(
        srcId, dstId,
        ComputeOp{ComputeOpSlice(m_controlBlock->symGraph->resolve(left),
                                 m_controlBlock->symGraph->resolve(right),
                                 m_controlBlock->symGraph->resolve(top),
                                 m_controlBlock->symGraph->resolve(bottom))});
    return Tensor{dstId, m_controlBlock};
  }

  void output(const Tensor &src) {
    assert(m_controlBlock->output ==
           details::ComputeGraphControlBlock::NullNode);
    m_controlBlock->output = src.m_nodeId;
    m_controlBlock->symGraph->debugDump();
  }

  void setLayout(const Tensor &tensor, std::optional<ActivationLayout> layout) {
    m_controlBlock->hypergraph.get(tensor.m_nodeId).m_layout = layout;
  }

  void setType(const Tensor &tensor, std::optional<FloatType> type) {
    m_controlBlock->hypergraph.get(tensor.m_nodeId).m_type = type;
  }

  Tensor Conv3x3(const Tensor &src, unsigned int K, bool bias = true) {
    return conv2d(src, glm::uvec2{3, 3}, K, bias, glm::uvec2{1, 1},
                  glm::uvec2{1, 1});
  }

  Tensor ReLU(const Tensor &src) {
    return activation(src, ActivationFunction::ReLU);
  }

  Tensor MaxPool(const Tensor &src, glm::uvec2 kernelSize,
                 std::optional<glm::uvec2> padding = std::nullopt,
                 std::optional<glm::uvec2> stride = std::nullopt) {
    glm::uvec2 p = padding.value_or(glm::uvec2(0, 0));
    glm::uvec2 s = stride.value_or(kernelSize);
    return pool(src, kernelSize, p, s, PoolFunction::Max);
  }

  Tensor NearestUpsample(const Tensor &src, unsigned int scalingFactor) {
    return upsample(src, scalingFactor, FilterMode::Nearest);
  }

  // hypergraph::ConstGraph<ComputeTensor, ComputeOp> freeze() const {
  //   return hypergraph::ConstGraph<ComputeTensor, ComputeOp>(
  //       m_controlBlock->hypergraph);
  // }

private:
  std::shared_ptr<details::ComputeGraphControlBlock> m_controlBlock;
};

} // namespace vkcnn
