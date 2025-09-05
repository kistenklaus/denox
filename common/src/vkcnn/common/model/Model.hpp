#pragma once

#include "vkcnn/common/AutoPadMode.hpp"
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
#include "vkcnn/common/tensor/BiasHostTensor.hpp"
#include "vkcnn/common/tensor/FilterHostTensor.hpp"
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
        symGraph(std::make_shared<vkcnn::SymGraph>()) {}

  vkcnn::hypergraph::NodeId input;
  vkcnn::hypergraph::NodeId output;
  vkcnn::hypergraph::AdjGraph<ComputeTensor, ComputeOp> hypergraph;

  std::shared_ptr<vkcnn::SymGraph> symGraph;
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

  std::uint64_t id() const { return static_cast<std::uint64_t>(m_nodeId); }
  Tensor() : m_nodeId(hypergraph::NodeId(0)), m_controlBlock(nullptr) {}

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
               std::optional<FloatType> type = std::nullopt,
               std::optional<Sym> W = std::nullopt,
               std::optional<Sym> H = std::nullopt) const {
    Sym w = Sym::Const(0);
    if (W.has_value()) {
      w = *W;
    } else {
      w = m_controlBlock->symGraph->var();
    }
    Sym h = Sym::Const(0);
    if (H.has_value()) {
      h = *H;
    } else {
      h = m_controlBlock->symGraph->var();
    }

    SymTensorExtent extent{w, h};
    assert(m_controlBlock->input ==
           details::ComputeGraphControlBlock::NullNode);
    hypergraph::NodeId id =
        m_controlBlock->hypergraph.emplaceNode(extent, channels, layout, type);
    m_controlBlock->input = id;
    return Tensor{id, m_controlBlock};
  }

  Tensor conv2d(const Tensor &src, vkcnn::FilterHostTensorConstView W,
                std::optional<vkcnn::BiasHostTensorConstView> B,
                AutoPadMode autoPad, glm::uvec2 stride,
                std::optional<glm::uvec2> padding, glm::uvec2 dilation,
                std::optional<FloatType> atype = std::nullopt) const {
    // We still don’t support dilation ≠ 1 in the backend
    if (dilation != glm::uvec2(1, 1)) {
      throw std::runtime_error(
          "vkcnn: conv2d currently does not support dilation != (1,1)");
    }

    // Fetch input node & kernel size (sx = width, sy = height)
    hypergraph::NodeId srcId = src.m_nodeId;
    const auto &srcTensor = m_controlBlock->hypergraph.get(srcId);
    glm::uvec2 kernelSize = glm::uvec2(W.shape().s, W.shape().r);

    // Resolve padding based on autoPad
    glm::uvec2 pad{0, 0};

    switch (autoPad) {
    case AutoPadMode::None: {
      if (!padding.has_value()) {
        throw std::runtime_error(
            "vkcnn: conv2d(autoPad=None) requires explicit padding");
      }
      pad = *padding;
      break;
    }

    case AutoPadMode::Zero: {
      pad = glm::uvec2(0, 0);
      break;
    }

    case AutoPadMode::SameUpper:
    case AutoPadMode::SameLower: {
      if (stride != glm::uvec2(1, 1)) {
        throw std::runtime_error(
            "vkcnn: conv2d SAME_* only supported for stride=(1,1)");
      }

      const unsigned sumX = (kernelSize.x > 0) ? (kernelSize.x - 1u) : 0u;
      const unsigned sumY = (kernelSize.y > 0) ? (kernelSize.y - 1u) : 0u;

      if ((sumX & 1u) != 0u || (sumY & 1u) != 0u) {
        throw std::runtime_error(
            "vkcnn: conv2d SAME_* requires symmetric padding; (kernel-1) must "
            "be even in both dims");
      }

      pad.x = sumX / 2u;
      pad.y = sumY / 2u;
      break;
    }

    default:
      throw std::runtime_error("vkcnn: conv2d received unknown AutoPadMode");
    }

    SymTensorExtent extent{
        m_controlBlock->symGraph->pool(srcTensor.m_extent.width, kernelSize.x,
                                       pad.x, stride.x, dilation.x, true),
        m_controlBlock->symGraph->pool(srcTensor.m_extent.height, kernelSize.y,
                                       pad.y, stride.y, dilation.y, true)};

    if (autoPad == AutoPadMode::SameUpper ||
        autoPad == AutoPadMode::SameLower) {
      auto &g = m_controlBlock->symGraph;
      const Sym outW = g->resolve(extent.width);
      const Sym outH = g->resolve(extent.height);
      const Sym inW = g->resolve(srcTensor.m_extent.width);
      const Sym inH = g->resolve(srcTensor.m_extent.height);

      if (!(outW == inW && outH == inH)) {
        throw std::runtime_error("vkcnn: conv2d SAME_* rejected: shape is not "
                                 "provably preserved with symmetric padding");
      }
    }

    hypergraph::NodeId dstId = m_controlBlock->hypergraph.emplaceNode(
        extent, W.shape().k, std::nullopt, std::nullopt);

    std::optional<vkcnn::BiasHostTensor> b = std::nullopt;
    if (B.has_value()) {
      b.emplace(*B);
    }

    m_controlBlock->hypergraph.addEdge(
        srcId, dstId,
        ComputeOp{ComputeOpConv(vkcnn::FilterHostTensor{W}, std::move(b), pad,
                                stride, atype)});

    return Tensor{dstId, m_controlBlock};
  }

  Tensor conv2d(const Tensor &src, vkcnn::FilterHostTensorConstView W,
                std::optional<vkcnn::BiasHostTensorConstView> B,
                glm::uvec2 stride, glm::uvec2 padding, glm::uvec2 dilation,
                std::optional<FloatType> atype = std::nullopt) const {
    if (dilation != glm::uvec2(1, 1)) {
      throw std::runtime_error(
          "vkcnn: Does currently not support dilation != (1,1)");
    }
    hypergraph::NodeId srcId = src.m_nodeId;
    const auto &srcTensor = m_controlBlock->hypergraph.get(srcId);
    glm::uvec2 kernelSize = glm::uvec2(W.shape().s, W.shape().r);
    SymTensorExtent extent{
        m_controlBlock->symGraph->pool(srcTensor.m_extent.width, kernelSize.x,
                                       padding.x, stride.x, dilation.x, true),
        m_controlBlock->symGraph->pool(srcTensor.m_extent.height, kernelSize.y,
                                       padding.y, stride.y, dilation.y, true)};
    hypergraph::NodeId dstId = m_controlBlock->hypergraph.emplaceNode(
        extent, W.shape().k, std::nullopt, std::nullopt);

    std::optional<vkcnn::BiasHostTensor> b = std::nullopt;
    if (B.has_value()) {
      b.emplace(*B);
    }

    m_controlBlock->hypergraph.addEdge(
        srcId, dstId,
        ComputeOp{ComputeOpConv(vkcnn::FilterHostTensor{W}, std::move(b),
                                padding, stride, atype)});
    return Tensor{dstId, m_controlBlock};
  }

  Tensor activation(const Tensor &src, ActivationFunction func) const {
    hypergraph::NodeId srcId = src.m_nodeId;
    const auto &srcNode = m_controlBlock->hypergraph.get(srcId);

    hypergraph::NodeId dstId = m_controlBlock->hypergraph.emplaceNode(
        srcNode.m_extent, srcNode.m_channels, std::nullopt, std::nullopt);

    m_controlBlock->hypergraph.addEdge(srcId, dstId,
                                       ComputeOp{ComputeOpActivation{func}});
    return Tensor{dstId, m_controlBlock};
  }

  Tensor upsample(const Tensor &src, unsigned int scalingFactor,
                  FilterMode mode) const {
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
              glm::uvec2 stride, glm::uvec2 dilation,
              PoolFunction poolFunc) const {
    if (dilation != glm::uvec2(1, 1)) {
      throw std::runtime_error(
          "vkcnn: Model::pool, does not support dilation != (1,1).");
    }

    hypergraph::NodeId srcId = src.m_nodeId;
    const auto &srcNode = m_controlBlock->hypergraph.get(srcId);
    SymTensorExtent extent{
        m_controlBlock->symGraph->pool(srcNode.m_extent.width, kernelSize.x,
                                       padding.x, stride.x, dilation.x),
        m_controlBlock->symGraph->pool(srcNode.m_extent.height, kernelSize.y,
                                       padding.y, stride.y, dilation.y)};

    hypergraph::NodeId dstId = m_controlBlock->hypergraph.emplaceNode(
        extent, srcNode.m_channels, std::nullopt, std::nullopt);

    m_controlBlock->hypergraph.addEdge(
        srcId, dstId,
        ComputeOp{ComputeOpPool(kernelSize, padding, stride, poolFunc)});
    return Tensor{dstId, m_controlBlock};
  }

  Tensor concat(const Tensor &src0, const Tensor &src1) const {
    hypergraph::NodeId src0Id = src0.m_nodeId;
    auto src0Node = m_controlBlock->hypergraph.get(src0Id);

    hypergraph::NodeId src1Id = src1.m_nodeId;
    auto src1Node = m_controlBlock->hypergraph.get(src1Id);

    if (src0Node.m_extent != src1Node.m_extent) {
      throw std::runtime_error(
          "Model::concat: Failed to prove that "
          "spatial dims of concat arguments match.\nvkcnn only accepts concat "
          "operations, if it can prove that all arguments have the same "
          "spatial extent. \nMake sure that your inputs are either cropped "
          "before passing them to concat,\nor align the network inputs spatial "
          "dimensions such that all concat arguments are provably equal.");
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
             PaddingMode mode) const {

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
  Tensor slice(const Tensor &src0, L left, R right, T top, B bottom) const {

    auto srcId = src0.m_nodeId;
    auto srcNode = m_controlBlock->hypergraph.get(srcId);

    SymTensorExtent extent{
        m_controlBlock->symGraph->sub(right, left),
        m_controlBlock->symGraph->sub(bottom, top),
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

  void output(const Tensor &src) const {
    assert(m_controlBlock->output ==
           details::ComputeGraphControlBlock::NullNode);
    m_controlBlock->output = src.m_nodeId;
  }

  void setLayout(const Tensor &tensor,
                 std::optional<ActivationLayout> layout) const {
    m_controlBlock->hypergraph.get(tensor.m_nodeId).m_layout = layout;
  }

  void setType(const Tensor &tensor, std::optional<FloatType> type) const {
    m_controlBlock->hypergraph.get(tensor.m_nodeId).m_type = type;
  }

  Tensor ReLU(const Tensor &src) const {
    return activation(src, ActivationFunction::ReLU);
  }

  Tensor LeakyReLU(const Tensor &src, float alpha = 0.01) const {
    if (std::abs(alpha - 0.01) > 1e-8) {
      throw std::runtime_error("Model::LeakyReLU not really implemented, only "
                               "works with alpha=0.01.");
    }
    return activation(src, ActivationFunction::LeakyReLU);
  }

  Tensor MaxPool(const Tensor &src, glm::uvec2 kernelSize,
                 std::optional<glm::uvec2> padding = std::nullopt,
                 std::optional<glm::uvec2> stride = std::nullopt,
                 std::optional<glm::uvec2> dilation = std::nullopt) const {
    glm::uvec2 p = padding.value_or(glm::uvec2(0, 0));
    glm::uvec2 s = stride.value_or(kernelSize);
    glm::uvec2 d = dilation.value_or(glm::uvec2(1, 1));
    return pool(src, kernelSize, p, s, d, PoolFunction::Max);
  }

  Tensor NearestUpsample(const Tensor &src, unsigned int scalingFactor) const {
    return upsample(src, scalingFactor, FilterMode::Nearest);
  }

  const vkcnn::hypergraph::AdjGraph<ComputeTensor, ComputeOp> &graph() const {
    return m_controlBlock->hypergraph;
  }

  vkcnn::SymGraph symGraph() { return *m_controlBlock->symGraph; }

  static Model import(std::string_view path);

private:
  std::shared_ptr<details::ComputeGraphControlBlock> m_controlBlock;
};

} // namespace vkcnn
