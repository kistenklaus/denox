#pragma once

#include "algorithm/pattern_matching/EdgePattern.fwd.hpp"
#include "algorithm/pattern_matching/GraphPattern.hpp"
#include "memory/container/vector.hpp"
#include "memory/dtype/dtype.hpp"
#include "memory/hypergraph/ConstGraph.hpp"
#include "memory/tensor/ActivationLayout.hpp"
#include "shaders/IShader.hpp"
namespace denox::compiler::shaders {

class DirectConvShader : public compiler::IShader {
private:
  using Pattern = algorithm::GraphPattern<TensorInstance, ComputeOp>;

  static constexpr unsigned int CONV_PATTERN = 0;
  static constexpr unsigned int CONV_RELU_PATTERN = 1;

public:
  DirectConvShader() {
    const auto tensorSupported = [](const TensorInstance &tensor) {
      if (tensor.type != memory::Dtype::F16) {
        return false;
      }
      if (tensor.layout != memory::ActivationLayout::HWC &&
          tensor.layout != memory::ActivationLayout::HWC8 &&
          tensor.layout != memory::ActivationLayout::CHWC8) {
        return false;
      }
      return true;
    };

    {
      Pattern conv_pattern;
      auto in = conv_pattern.matchNode();
      auto op = in->matchOutgoing();
      op->matchRank(1);
      op->matchValue(
          [](const ComputeOp &op) { return op.tag() == ComputeOpTag::Conv; });
      auto out = op->matchDst();

      in->matchValue(tensorSupported);
      out->matchValue(tensorSupported);

      m_convPattern.emplace_back(std::move(op));
      m_capabilities.patterns.emplace_back(std::move(conv_pattern),
                                           std::move(in), std::move(out));
    }
    { // possibly more patterns.
      Pattern conv_relu_pattern;
      auto in = conv_relu_pattern.matchNode();
      auto conv = in->matchOutgoing();
      auto inter = conv->matchDst();
      auto relu = inter->matchOutgoing();
      auto out = relu->matchDst();

      conv->matchRank(1);
      conv->matchValue(
          [](const ComputeOp &op) { return op.tag() == ComputeOpTag::Conv; });
      relu->matchRank(1);
      relu->matchValue([](const ComputeOp &op) {
        return op.tag() == ComputeOpTag::Activation;
      });

      in->matchValue(tensorSupported);
      out->matchValue(tensorSupported);

      m_convPattern.emplace_back(std::move(conv));
      m_capabilities.patterns.emplace_back(std::move(conv_relu_pattern),
                                           std::move(in), std::move(out));
    }
  }

  const ShaderCapabilities &capabilities() const final override {
    return m_capabilities;
  }

  std::size_t
  parameterMemorySize(const memory::ConstGraph<TensorInstance, ComputeOp> &graph,
                      unsigned int pattern,
                      const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
                          &match) const final override {
    const auto &convPattern = m_convPattern[pattern];
    memory::EdgeId convId = match[convPattern];
    const ComputeOp &op = graph.get(convId);
    assert(op.tag() == ComputeOpTag::Conv);
    const auto &conv = op.conv();
    std::size_t elemCount = conv->W->shape().elemCount();
    if (conv->B != nullptr) {
      elemCount += conv->B->shape();
    }
    return elemCount * memory::Dtype::F16.size();
  }

  void implement(
      [[maybe_unused]] unsigned int pattern,
      [[maybe_unused]] const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
          &match) const final override {}

  memory::string name() const final override { return "direct-conv"; }

private:
  ShaderCapabilities m_capabilities;
  memory::vector<algorithm::EdgePatternHandle<TensorInstance, ComputeOp>>
      m_convPattern;
};

} // namespace denox::compiler::shaders
