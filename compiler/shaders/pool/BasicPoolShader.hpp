#pragma once

#include "algorithm/pattern_matching/GraphPattern.hpp"
#include "model/PoolFunction.hpp"
#include "shaders/IShader.hpp"
namespace denox::compiler::shaders {

class BasicPoolShader final : public IShader {
  static constexpr std::size_t HWC_HWC_F16_PATTERN = 0;
  static constexpr std::size_t CHWC8_CHWC8_F16_PATTERN = 1;

public:
  using Pattern = algorithm::GraphPattern<TensorInstance, ComputeOp>;

  BasicPoolShader() {
    {
      Pattern hwc_hwc_f16;
      auto in = hwc_hwc_f16.matchNode();
      auto pool = in->matchOutgoing();
      auto out = pool->matchDst();

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

      pool->matchRank(1);
      pool->matchValue([](const ComputeOp &op) {
        if (op.tag() != ComputeOpTag::Pool) {
          return false;
        }
        const auto &pool = op.pool();
        if (pool->func != PoolFunction::Max) {
          return false;
        }
        if (pool->stride != pool->kernelSize) {
          return false;
        }
        if (pool->padding != memory::uvec2(0, 0)) {
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
      auto pool = in->matchOutgoing();
      auto out = pool->matchDst();

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

      pool->matchRank(1);
      pool->matchValue([](const ComputeOp &op) {
        if (op.tag() != ComputeOpTag::Pool) {
          return false;
        }
        const auto &pool = op.pool();
        if (pool->func != PoolFunction::Max) {
          return false;
        }
        if (pool->stride != pool->kernelSize) {
          return false;
        }
        if (pool->padding != memory::uvec2(0, 0)) {
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

  memory::string name([[maybe_unused]] unsigned int pattern) const final override { return "basic-pool"; }

private:
  ShaderCapabilities m_capabilities;
};

} // namespace denox::compiler::shaders
