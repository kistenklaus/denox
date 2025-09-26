#pragma once

#include "algorithm/pattern_matching/GraphPattern.hpp"
#include "model/FilterMode.hpp"
#include "shaders/IShader.hpp"
namespace denox::compiler::shaders {

class BasicUpsampleShader : public IShader {
public:
  using Pattern = algorithm::GraphPattern<TensorInstance, ComputeOp>;

  BasicUpsampleShader() {
    const auto supportedTensor = [](const TensorInstance &tensor) {
      if (tensor.type != memory::Dtype::F16) {
        return false;
      }
      return tensor.layout == memory::ActivationLayout::HWC ||
             tensor.layout == memory::ActivationLayout::HWC8 ||
             tensor.layout == memory::ActivationLayout::CHWC8;
    };
    {
      Pattern upsamplePattern;
      auto in = upsamplePattern.matchNode();
      auto upsample = in->matchOutgoing();
      auto out = upsample->matchDst();

      in->matchValue(supportedTensor);
      out->matchValue(supportedTensor);

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
      m_patternHandles.emplace_back(in, std::move(upsample), out);
      m_capabilities.patterns.emplace_back(std::move(upsamplePattern),
                                           std::move(in), std::move(out));
    }
  }

  memory::optional<unsigned int>
  acceptMatch(const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
              unsigned int pattern,
              const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
                  &match) const final override {
    const auto &patternHandles = m_patternHandles[pattern];
    const auto &in = opGraph.get(match[patternHandles.in]);
    const auto &out = opGraph.get(match[patternHandles.out]);
    if (in.layout != out.layout) {
      return memory::nullopt;
    }
    return pattern;
  }

  const ShaderCapabilities &capabilities() const final override {
    return m_capabilities;
  }

  // TODO Figure out the return from here, maybe directly somethig like a
  // dispatch with a compiled SPIR-V or something like this.
  void implement(Impl &impl,
                 const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
                 [[maybe_unused]] unsigned int pattern,
                 [[maybe_unused]] const algorithm::ConstGraphMatch<
                     TensorInstance, ComputeOp> &match) const final override {
    const auto &patternHandles = m_patternHandles[pattern];
    memory::NodeId inId = match[patternHandles.in];
    memory::NodeId outId = match[patternHandles.out];

    auto dispatch = impl.dispatch({});
    dispatch.addBinding(inId);
    dispatch.addBinding(outId);
    const auto &in = opGraph.get(inId);
    dispatch.addPushConstant(PushConstant::Dynamic(in.extent.x));
    dispatch.addPushConstant(PushConstant::Dynamic(in.extent.y));
    dispatch.setName(name(pattern));
    dispatch.setSourcePath(m_srcPath);
  }

  memory::string
  name([[maybe_unused]] unsigned int pattern) const final override {
    return "basic-upsample";
  }

private:
  ShaderCapabilities m_capabilities;
  struct Handles {
    Pattern::NP in;
    Pattern::EP upsample;
    Pattern::NP out;
  };
  memory::vector<Handles> m_patternHandles;

  io::Path m_srcPath =
      io::Path::cwd() / "compiler/shaders/upsample/basic_upsample.comp";
};

} // namespace denox::compiler::shaders
