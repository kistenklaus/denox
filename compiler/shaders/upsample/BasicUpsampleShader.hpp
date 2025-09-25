#pragma once

#include "algorithm/pattern_matching/GraphPattern.hpp"
#include "model/FilterMode.hpp"
#include "shaders/IShader.hpp"
namespace denox::compiler::shaders {

class BasicUpsampleShader : public IShader {
public:
  using Pattern = algorithm::GraphPattern<TensorInstance, ComputeOp>;

  BasicUpsampleShader() {
    {
      Pattern hwc_hwc_f16;
      auto in = hwc_hwc_f16.matchNode();
      auto upsample = in->matchOutgoing();
      auto out = upsample->matchDst();

      in->matchValue([](const TensorInstance &tensor) {
        if (tensor.type != memory::Dtype::F16) {
          return false;
        }
        return tensor.layout == memory::ActivationLayout::HWC ||
               tensor.layout == memory::ActivationLayout::HWC8;
      });
      out->matchValue([](const TensorInstance &tensor) {
        if (tensor.type != memory::Dtype::F16) {
          return false;
        }
        return tensor.layout == memory::ActivationLayout::HWC ||
               tensor.layout == memory::ActivationLayout::HWC8;
      });

      upsample->matchRank(1);
      upsample->matchValue([](const ComputeOp &op) {
        if (op.tag() != ComputeOpTag::Upsample) {
          return false;
        }
        const auto &upsample = op.upsample();
        if (upsample.mode != FilterMode::Nearest) {
          return false;
        }
        return true;
      });
      m_capabilities.patterns.emplace_back(std::move(hwc_hwc_f16),
                                           std::move(in), std::move(out));
    }
    {
      Pattern chwc8_chwc8_f16;
      auto in = chwc8_chwc8_f16.matchNode();
      auto upsample = in->matchOutgoing();
      auto out = upsample->matchDst();

      in->matchValue([](const TensorInstance &tensor) {
        if (tensor.type != memory::Dtype::F16) {
          return false;
        }
        return tensor.layout == memory::ActivationLayout::CHWC8;
      });
      out->matchValue([](const TensorInstance &tensor) {
        if (tensor.type != memory::Dtype::F16) {
          return false;
        }
        return tensor.layout == memory::ActivationLayout::CHWC8;
      });

      upsample->matchRank(1);
      upsample->matchValue([](const ComputeOp &op) {
        if (op.tag() != ComputeOpTag::Upsample) {
          return false;
        }
        const auto &upsample = op.upsample();
        if (upsample.mode != FilterMode::Nearest) {
          return false;
        }
        return true;
      });
      m_capabilities.patterns.emplace_back(std::move(chwc8_chwc8_f16),
                                           std::move(in), std::move(out));
    }
  }

  const ShaderCapabilities &capabilities() const final override {
    return m_capabilities;
  }

  // TODO Figure out the return from here, maybe directly somethig like a
  // dispatch with a compiled SPIR-V or something like this.
  void implement(
      [[maybe_unused]] unsigned int pattern,
      [[maybe_unused]] const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
          &match) const final override {}

  memory::string name() const final override { return "basic-upsample"; }

private:
  ShaderCapabilities m_capabilities;
};

} // namespace denox::compiler::shaders
