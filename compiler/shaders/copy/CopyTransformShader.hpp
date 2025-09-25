#pragma once

#include "algorithm/pattern_matching/GraphPattern.hpp"
#include "shaders/IShader.hpp"
namespace denox::compiler::shaders {

class CopyTransformShader : public IShader {
public:
  using Pattern = algorithm::GraphPattern<ComputeTensor, ComputeOp>;

  CopyTransformShader() {
    {
      Pattern hwc_hwc_hwc_explicitConcat;
      auto concat = hwc_hwc_hwc_explicitConcat.matchEdge();
      concat->matchRank(2);
      auto in0 = concat->matchSrc(0);
      auto in1 = concat->matchSrc(1);
      auto out = concat->matchDst();

      const auto hwcLayout = [](const ComputeTensor &tensor) {
        return tensor.layout() == memory::ActivationLayout::HWC ||
               tensor.layout() == memory::ActivationLayout::HWC8;
      };
      in0->matchValue(hwcLayout);
      in1->matchValue(hwcLayout);
      out->matchValue(hwcLayout);

      m_capabilities.patterns.emplace_back(
          std::move(hwc_hwc_hwc_explicitConcat), std::move(in0), std::move(in1),
          std::move(out));
    }

    {
      Pattern chw8_chwc8_chwc8_explicitConcat;
      auto concat = chw8_chwc8_chwc8_explicitConcat.matchEdge();
      concat->matchRank(2);
      auto in0 = concat->matchSrc(0);
      auto in1 = concat->matchSrc(1);
      auto out = concat->matchDst();

      const auto chwc8Layout = [](const ComputeTensor &tensor) {
        return tensor.layout() == memory::ActivationLayout::CHWC8;
      };
      in0->matchValue(chwc8Layout);
      in1->matchValue(chwc8Layout);
      out->matchValue(chwc8Layout);

      m_capabilities.patterns.emplace_back(
          std::move(chw8_chwc8_chwc8_explicitConcat), std::move(in0),
          std::move(in1), std::move(out));
    }
  }

  const ShaderCapabilities &capabilities() const final override {
    return m_capabilities;
  }

  // TODO Figure out the return from here, maybe directly somethig like a
  // dispatch with a compiled SPIR-V or something like this.
  void implement([[maybe_unused]] unsigned int pattern,
                 [[maybe_unused]] const algorithm::ConstGraphMatch<ComputeTensor, ComputeOp>
                     &match) const final override {}

  memory::string name() const final override {
    return "copy-transform";
  }

private:
  ShaderCapabilities m_capabilities;
};

} // namespace denox::compiler::shaders
