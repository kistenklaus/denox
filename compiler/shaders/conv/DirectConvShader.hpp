#pragma once

#include "algorithm/pattern_matching/GraphPattern.hpp"
#include "memory/dtype/dtype.hpp"
#include "memory/tensor/ActivationLayout.hpp"
#include "shaders/IShader.hpp"
namespace denox::compiler::shaders {

class DirectConvShader : public compiler::IShader {
private:
  using Pattern = algorithm::GraphPattern<ComputeTensor, ComputeOp>;

public:
  DirectConvShader() {
    const auto tensorSupported = [](const ComputeTensor &tensor) {
      if (tensor.type() != memory::Dtype::F16) {
        return false;
      }
      if (tensor.layout() != memory::ActivationLayout::HWC &&
          tensor.layout() != memory::ActivationLayout::HWC8 &&
          tensor.layout() != memory::ActivationLayout::CHWC8) {
        return false;
      }
      return true;
    };

    {
      Pattern any_any_f16;
      auto in = any_any_f16.matchNode();
      auto op = in->matchOutgoing();
      op->matchRank(1);
      op->matchValue(
          [](const ComputeOp &op) { return op.tag() == ComputeOpTag::Conv; });
      auto out = op->matchDst();

      in->matchValue(tensorSupported);
      out->matchValue(tensorSupported);

      m_capabilities.patterns.emplace_back(std::move(any_any_f16),
                                           std::move(in), std::move(out));
    }
    { // possibly more patterns.
      Pattern pattern1;
      auto in = pattern1.matchNode();
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

      m_capabilities.patterns.emplace_back(std::move(pattern1), std::move(in),
                                           std::move(out));
    }
  }

  const ShaderCapabilities &capabilities() const final override {
    return m_capabilities;
  }

  void implement([[maybe_unused]] unsigned int pattern,
                 [[maybe_unused]] const algorithm::ConstGraphMatch<
                     ComputeTensor, ComputeOp> &match) const final override {}

private:
  ShaderCapabilities m_capabilities;
};

} // namespace denox::compiler::shaders
