#include "denox/compiler/implement/shaders/noop/NoOp.hpp"
#include "denox/common/TensorFormat.hpp"
#include "denox/common/TensorStorage.hpp"
#include "denox/compiler/frontend/model/ComputeOp.hpp"

namespace denox::compiler {

NoOp::NoOp() {
  {
    Pattern pattern;
    auto in = pattern.matchNode();
    auto noop = in->matchOutgoing();
    auto out = noop->matchDst();
    noop->matchRank(1);
    noop->matchValue(
        [](const ComputeOp &op) { return op.tag() == ComputeOpKind::None; });

    m_handles.in = in;
    m_handles.out = in;
    m_capabilities.patterns.emplace_back(std::move(pattern), std::move(in),
                                         std::move(out));
  }
}

memory::vector<unsigned int> NoOp::acceptMatch(
    [[maybe_unused]] const memory::ConstGraph<TensorInstance, ComputeOp>
        &opGraph,
    [[maybe_unused]] unsigned int pattern,
    [[maybe_unused]] const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
        &match) const {
  const auto in = opGraph.get(match[m_handles.in]);
  const auto out = opGraph.get(match[m_handles.out]);

  auto allowed = [](auto inVal, auto outVal, auto optimal) {
    // invalid only if both are specialized and not equal
    return !(inVal != optimal && outVal != optimal && inVal != outVal);
  };

  if (!allowed(in.format, out.format, TensorFormat::Optimal)) {
    return {};
  }

  if (!allowed(in.storage, out.storage, TensorStorage::Optimal)) {
    return {};
  }

  return {0};
}

} // namespace denox::compiler
