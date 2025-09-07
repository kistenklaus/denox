#include "./Model.hpp"

namespace denox::compiler {

Tensor Model::input(unsigned int channels,
                    memory::optional<memory::ActivationLayout> layout,
                    memory::optional<memory::Dtype> type, memory::optional<Sym> W,
                    memory::optional<Sym> H) {
  Sym w = Sym::Const(0);
  if (W.has_value()) {
    w = *W;
  } else {
    w = m_controlBlock->symGraph.var();
  }
  Sym h = Sym::Const(0);
  if (H.has_value()) {
    h = *H;
  } else {
    h = m_controlBlock->symGraph.var();
  }

  sym_vec2 extent{w, h};
  assert(m_controlBlock->input == details::model::ModelControlBlock::NullNode);
  memory::NodeId id =
      m_controlBlock->hypergraph.emplaceNode(extent, channels, layout, type);
  m_controlBlock->input = id;
  return Tensor{id, m_controlBlock.get()};
}

Tensor Model::conv2d(const Tensor &src, memory::FilterTensorConstView W,
                     memory::optional<memory::BiasTensorConstView> B,
                     AutoPadMode autoPad, memory::uvec2 stride,
                     memory::optional<memory::uvec2> padding,
                     memory::uvec2 dilation,
                     memory::optional<memory::Dtype> atype) const {
  // We still don’t support dilation ≠ 1 in the backend
  if (dilation != memory::uvec2(1, 1)) {
    throw std::runtime_error(
        "vkcnn: conv2d currently does not support dilation != (1,1)");
  }

  // Fetch input node & kernel size (sx = width, sy = height)
  memory::NodeId srcId = src.m_nodeId;
  const auto &srcTensor = m_controlBlock->hypergraph.get(srcId);
  memory::uvec2 kernelSize = memory::uvec2(W.shape().s, W.shape().r);

  // Resolve padding based on autoPad
  memory::uvec2 pad{0, 0};

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
    pad = memory::uvec2(0, 0);
    break;
  }

  case AutoPadMode::SameUpper:
  case AutoPadMode::SameLower: {
    if (stride != memory::uvec2(1, 1)) {
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

  sym_vec2 extent{
      m_controlBlock->symGraph.pool(srcTensor.m_extent.x.asSym(), kernelSize.x,
                                    pad.x, stride.x, dilation.x, true),
      m_controlBlock->symGraph.pool(srcTensor.m_extent.y.asSym(), kernelSize.y,
                                    pad.y, stride.y, dilation.y, true)};

  if (autoPad == AutoPadMode::SameUpper || autoPad == AutoPadMode::SameLower) {
    auto &g = m_controlBlock->symGraph;
    const Sym outW = g.resolve(extent.x.asSym());
    const Sym outH = g.resolve(extent.y.asSym());
    const Sym inW = g.resolve(srcTensor.m_extent.x.asSym());
    const Sym inH = g.resolve(srcTensor.m_extent.y.asSym());

    if (!(outW == inW && outH == inH)) {
      throw std::runtime_error("vkcnn: conv2d SAME_* rejected: shape is not "
                               "provably preserved with symmetric padding");
    }
  }

  memory::NodeId dstId = m_controlBlock->hypergraph.emplaceNode(
      extent, W.shape().k, memory::nullopt, memory::nullopt);

  memory::optional<memory::BiasTensor> b = memory::nullopt;
  if (B.has_value()) {
    b.emplace(*B);
  }

  m_controlBlock->hypergraph.addEdge(
      srcId, dstId,
      ComputeOp{ComputeOpConv(memory::FilterTensor{W}, std::move(b), pad,
                              stride, atype)});

  return Tensor{dstId, m_controlBlock.get()};
}

Tensor Model::activation(const Tensor &src, ActivationFunction func) const {
  memory::NodeId srcId = src.m_nodeId;
  const auto &srcNode = m_controlBlock->hypergraph.get(srcId);

  memory::NodeId dstId = m_controlBlock->hypergraph.emplaceNode(
      srcNode.m_extent, srcNode.m_channels, memory::nullopt, memory::nullopt);

  m_controlBlock->hypergraph.addEdge(srcId, dstId,
                                     ComputeOp{ComputeOpActivation{func}});
  return Tensor{dstId, m_controlBlock.get()};
}

Tensor Model::upsample(const Tensor &src, unsigned int scalingFactor,
                       FilterMode mode) const {
  memory::NodeId srcId = src.m_nodeId;
  const auto &srcNode = m_controlBlock->hypergraph.get(srcId);

  sym_vec2 extent{
      m_controlBlock->symGraph.mul(scalingFactor, srcNode.m_extent.x.asSym()),
      m_controlBlock->symGraph.mul(scalingFactor, srcNode.m_extent.y.asSym())};

  memory::NodeId dstId = m_controlBlock->hypergraph.emplaceNode(
      extent, srcNode.m_channels, memory::nullopt, memory::nullopt);

  m_controlBlock->hypergraph.addEdge(
      srcId, dstId, ComputeOp{ComputeOpUpsample{scalingFactor, mode}});
  return Tensor{dstId, m_controlBlock.get()};
}

Tensor Model::pool(const Tensor &src, memory::uvec2 kernelSize,
                   memory::uvec2 padding, memory::uvec2 stride,
                   memory::uvec2 dilation, PoolFunction poolFunc) const {
  if (dilation != memory::uvec2(1, 1)) {
    throw std::runtime_error(
        "vkcnn: Model::pool, does not support dilation != (1,1).");
  }

  memory::NodeId srcId = src.m_nodeId;
  const auto &srcNode = m_controlBlock->hypergraph.get(srcId);
  sym_vec2 extent{
      m_controlBlock->symGraph.pool(srcNode.m_extent.x.asSym(), kernelSize.x,
                                    padding.x, stride.x, dilation.x),
      m_controlBlock->symGraph.pool(srcNode.m_extent.y.asSym(), kernelSize.y,
                                    padding.y, stride.y, dilation.y)};

  memory::NodeId dstId = m_controlBlock->hypergraph.emplaceNode(
      extent, srcNode.m_channels, memory::nullopt, memory::nullopt);

  m_controlBlock->hypergraph.addEdge(
      srcId, dstId,
      ComputeOp{ComputeOpPool(kernelSize, padding, stride, poolFunc)});
  return Tensor{dstId, m_controlBlock.get()};
}

Tensor Model::concat(const Tensor &src0, const Tensor &src1) const {
  memory::NodeId src0Id = src0.m_nodeId;
  auto src0Node = m_controlBlock->hypergraph.get(src0Id);

  memory::NodeId src1Id = src1.m_nodeId;
  auto src1Node = m_controlBlock->hypergraph.get(src1Id);

  if (m_controlBlock->symGraph.resolve(src0Node.m_extent.x.asSym()) !=
          m_controlBlock->symGraph.resolve(src1Node.m_extent.x.asSym()) &&
      m_controlBlock->symGraph.resolve(src0Node.m_extent.y.asSym()) !=
          m_controlBlock->symGraph.resolve(src1Node.m_extent.y.asSym())) {
    throw std::runtime_error(
        "Model::concat: Failed to prove that "
        "spatial dims of concat arguments match.\nvkcnn only accepts concat "
        "operations, if it can prove that all arguments have the same "
        "spatial extent. \nMake sure that your inputs are either cropped "
        "before passing them to concat,\nor align the network inputs spatial "
        "dimensions such that all concat arguments are provably equal.");
  }

  memory::NodeId dstId = m_controlBlock->hypergraph.emplaceNode(
      src0Node.m_extent, src0Node.m_channels + src1Node.m_channels,
      memory::nullopt, memory::nullopt);

  m_controlBlock->hypergraph.addEdge(src0Id, src1Id, dstId,
                                     ComputeOp{ComputeOpConcat{}});

  return Tensor{dstId, m_controlBlock.get()};
}

Tensor Model::pad(const Tensor &src0, Sym left, Sym right, Sym top, Sym bottom,
                  PaddingMode mode) const {

  auto srcId = src0.m_nodeId;
  auto srcNode = m_controlBlock->hypergraph.get(srcId);

  sym_vec2 extent{
      m_controlBlock->symGraph.add(
          srcNode.m_extent.x.asSym(),
          m_controlBlock->symGraph.add(left, right, false)),
      m_controlBlock->symGraph.add(
          srcNode.m_extent.y.asSym(),
          m_controlBlock->symGraph.add(top, bottom, false)),
  };
  auto dstId = m_controlBlock->hypergraph.emplaceNode(
      extent, srcNode.channels(), memory::nullopt, memory::nullopt);

  m_controlBlock->hypergraph.addEdge(
      srcId, dstId,
      ComputeOp{ComputeOpPad(m_controlBlock->symGraph.resolve(left),
                             m_controlBlock->symGraph.resolve(right),
                             m_controlBlock->symGraph.resolve(top),
                             m_controlBlock->symGraph.resolve(bottom), mode)});
  return Tensor{dstId, m_controlBlock.get()};
}

Tensor Model::slice(const Tensor &src0, Sym left, Sym right, Sym top,
                    Sym bottom) const {

  auto srcId = src0.m_nodeId;
  auto srcNode = m_controlBlock->hypergraph.get(srcId);

  sym_vec2 extent{
      m_controlBlock->symGraph.sub(right, left),
      m_controlBlock->symGraph.sub(bottom, top),
  };
  auto dstId = m_controlBlock->hypergraph.emplaceNode(
      extent, srcNode.channels(), memory::nullopt, memory::nullopt);

  m_controlBlock->hypergraph.addEdge(
      srcId, dstId,
      ComputeOp{ComputeOpSlice(m_controlBlock->symGraph.resolve(left),
                               m_controlBlock->symGraph.resolve(right),
                               m_controlBlock->symGraph.resolve(top),
                               m_controlBlock->symGraph.resolve(bottom))});
  return Tensor{dstId, m_controlBlock.get()};
}

void Model::output(const Tensor &src) const {
  assert(m_controlBlock->output == details::model::ModelControlBlock::NullNode);
  m_controlBlock->output = src.m_nodeId;
}

} // namespace denox::compiler
