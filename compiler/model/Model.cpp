#include "./Model.hpp"
#include "diag/unreachable.hpp"
#include "model/ModelControlBlock.hpp"
#include <fmt/format.h>
#include <stdexcept>

namespace denox::compiler {

Tensor Model::input(unsigned int channels,
    const std::string& name,
                    memory::optional<memory::ActivationLayout> layout,
                    memory::optional<memory::Dtype> type,
                    memory::optional<Sym> W, memory::optional<Sym> H,
                    NamedExtent dynamicExtent) {
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
  m_controlBlock->inputExtentNames = dynamicExtent;
  m_controlBlock->inputName = name;
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
  const auto srcNode = m_controlBlock->hypergraph.get(srcId);

  memory::NodeId dstId = m_controlBlock->hypergraph.emplaceNode(
      srcNode.m_extent, srcNode.m_channels, memory::nullopt, memory::nullopt);

  m_controlBlock->hypergraph.addEdge(srcId, dstId,
                                     ComputeOp{ComputeOpActivation{func}});
  return Tensor{dstId, m_controlBlock.get()};
}

Tensor Model::upsample(const Tensor &src, unsigned int scalingFactor,
                       FilterMode mode) const {
  memory::NodeId srcId = src.m_nodeId;
  const auto srcNode = m_controlBlock->hypergraph.get(srcId);

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
  const auto srcNode = m_controlBlock->hypergraph.get(srcId);
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
  const auto srcNode = m_controlBlock->hypergraph.get(srcId);

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
  const auto srcNode = m_controlBlock->hypergraph.get(srcId);

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

void Model::output(const Tensor &src, const std::string& name, NamedExtent extentNames) const {
  // Currently we only support a single input / output.
  assert(m_controlBlock->output == details::model::ModelControlBlock::NullNode); 
  m_controlBlock->outputExtentNames = extentNames;
  m_controlBlock->output = src.m_nodeId;
  m_controlBlock->outputName = name;
}

Tensor Model::getInput() const {
  assert(m_controlBlock->input != details::model::ModelControlBlock::NullNode);
  return Tensor{m_controlBlock->input, m_controlBlock.get()};
}

Tensor Model::getOutput() const {
  assert(m_controlBlock->output != details::model::ModelControlBlock::NullNode);
  return Tensor{m_controlBlock->output, m_controlBlock.get()};
}

memory::string Model::to_string() const {
  memory::string str;
  if (m_controlBlock->meta.domain.has_value()) {
    str.append(fmt::format("- Domain: {}\n", *m_controlBlock->meta.domain));
  }
  if (m_controlBlock->meta.producerName.has_value() &&
      m_controlBlock->meta.producerVersion) {
    str.append(fmt::format("- Producer: {}@{}\n",
                           *m_controlBlock->meta.producerName,
                           *m_controlBlock->meta.producerVersion));
  }
  if (m_controlBlock->meta.modelVersion.has_value() &&
      m_controlBlock->meta.modelVersion) {
    str.append(
        fmt::format("- Version: {}\n", *m_controlBlock->meta.modelVersion));
  }
  auto input = getInput();
  str.append(fmt::format("- Input: (TensorID: {})\n    C: {}\n",
                         static_cast<std::uint64_t>(input.m_nodeId),
                         input.channels()));
  if (input.width().isSymbolic()) {
    str.append(fmt::format("    W: [{}] <- SymInt\n", (*input.width()).sym()));
  } else {
    str.append(fmt::format("    W: {}\n", (*input.width()).constant()));
  }
  if (input.height().isSymbolic()) {
    str.append(fmt::format("    H: [{}] <- SymInt\n", (*input.height()).sym()));
  } else {
    str.append(fmt::format("    H: {}\n", (*input.height()).constant()));
  }
  if (input.type().has_value()) {
    str.append(fmt::format("    dtype: {}\n", input.type()->to_string()));
  }

  if (input.layout().has_value()) {
    str.append(fmt::format("    layout: {}\n", input.layout()->to_string()));
  }

  auto output = getOutput();
  str.append(fmt::format("- Output: (TensorID: {})\n    C: {}\n",
                         static_cast<std::uint64_t>(output.m_nodeId),
                         output.channels()));
  if (output.width().isSymbolic()) {
    str.append(fmt::format("    W: [{}] <- SymInt\n", (*output.width()).sym()));
  } else {
    str.append(fmt::format("    W: {}\n", output.width().constant()));
  }
  if (output.height().isSymbolic()) {
    str.append(
        fmt::format("    H: [{}] <- SymInt\n", (*output.height()).sym()));
  } else {
    str.append(fmt::format("    H: {}\n", output.height().constant()));
  }
  if (output.type().has_value()) {
    str.append(fmt::format("    dtype: {}\n", output.type()->to_string()));
  }

  if (output.layout().has_value()) {
    str.append(fmt::format("    layout: {}\n", output.layout()->to_string()));
  }

  str.append("- Layers:\n");
  for (const auto edge : m_controlBlock->hypergraph.edges()) {
    const denox::compiler::ComputeOp &op =
        m_controlBlock->hypergraph.get(edge.id());

    auto srcs = edge.edge().src();
    std::string in;
    if (srcs.size() > 1) {
      in.append("(");
    }
    for (std::size_t s = 0; s < srcs.size(); ++s) {
      if (s != 0) {
        in.append(", ");
      }
      in.append(fmt::format("{}", static_cast<std::uint64_t>(srcs[s])));
    }
    if (srcs.size() > 1) {
      in.append(")");
    }

    std::string out =
        fmt::format("{}", static_cast<std::uint64_t>(edge.edge().dst()));

    std::string inout = fmt::format("{} -> {}", in, out);

    switch (op.tag()) {
    case ComputeOpTag::None:
      denox::compiler::diag::unreachable();
    case ComputeOpTag::Conv: {
      str.append(fmt::format("    Conv: {}\n", inout));
      const auto conv = op.conv();
      str.append(fmt::format("      - kernelSize: {}x{}\n", conv->W->shape().s,
                             conv->W->shape().r));
      str.append(fmt::format("      - stride:     ({},{})\n", conv->stride.x,
                             conv->stride.y));
      str.append(fmt::format("      - padding:    ({},{})\n", conv->padding.x,
                             conv->padding.y));
      str.append(fmt::format("      - bias:       {}\n", conv->B != nullptr));
      if (conv->atype.has_value()) {
        str.append(
            fmt::format("      - dtype:       {}\n", conv->atype->to_string()));
      }
      break;
    }
    case ComputeOpTag::Activation: {
      const auto acti = op.activation();
      switch (acti.func) {
      case ActivationFunction::ReLU:
        str.append(fmt::format("    ReLU: {}\n", inout));
        break;
      case ActivationFunction::LeakyReLU:
        str.append(fmt::format("    LeakyReLU: {}\n", inout));
        break;
      case ActivationFunction::SiLU:
        str.append(fmt::format("    SiLU: {}\n", inout));
        break;
      }
      break;
    }
    case ComputeOpTag::Upsample: {
      const auto upsample = op.upsample();
      str.append(fmt::format("    Upsample: {}\n", inout));
      switch (upsample.mode) {
      case FilterMode::Nearest:
        str.append("      - mode: Nearest\n");
        break;
      }
      str.append(
          fmt::format("      - scaling-factor: {}\n", upsample.scalingFactor));
      break;
    }
    case ComputeOpTag::Pool: {
      const auto &pool = op.pool();
      str.append(fmt::format("    Pool: {}\n", inout));
      str.append(fmt::format("      - kernelSize: {}x{}\n", pool->kernelSize.x,
                             pool->kernelSize.y));
      str.append(fmt::format("      - stride:     ({},{})\n", pool->stride.x,
                             pool->stride.y));
      str.append(fmt::format("      - padding:    ({},{})\n", pool->padding.x,
                             pool->padding.y));
      switch (pool->func) {
      case PoolFunction::Max:
        str.append("      - mode:       MaxPool\n");
        break;
      case PoolFunction::Avg:
        str.append("      - mode:       AveragePool\n");
        break;
      }
      break;
    }
    case ComputeOpTag::Concat: {
      str.append(fmt::format("    Concat: {}\n", inout));
      break;
    }
    case ComputeOpTag::Pad: {
      const auto &pad = op.pad();
      str.append(fmt::format("    Pad: {}\n", inout));
      str.append(fmt::format("      - left:       {}\n",
                             m_controlBlock->symGraph.to_string(pad->left)));
      str.append(fmt::format("      - right:      {}\n",
                             m_controlBlock->symGraph.to_string(pad->right)));
      str.append(fmt::format("      - top:        {}\n",
                             m_controlBlock->symGraph.to_string(pad->top)));
      str.append(fmt::format("      - bottom:     {}\n",
                             m_controlBlock->symGraph.to_string(pad->bottom)));
      break;
    }
    case ComputeOpTag::Slice: {
      const auto &slice = op.slice();
      str.append(fmt::format("    Slice: {}\n", inout));
      str.append(fmt::format("      - left:       {}\n",
                             m_controlBlock->symGraph.to_string(slice->left)));
      str.append(fmt::format("      - right:      {}\n",
                             m_controlBlock->symGraph.to_string(slice->right)));
      str.append(fmt::format("      - top:        {}\n",
                             m_controlBlock->symGraph.to_string(slice->top)));
      str.append(
          fmt::format("      - bottom:     {}\n",
                      m_controlBlock->symGraph.to_string(slice->bottom)));
      break;
    }
    }
  }

  return str;
}

} // namespace denox::compiler
