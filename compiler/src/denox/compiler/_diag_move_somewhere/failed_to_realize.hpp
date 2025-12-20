#pragma once

#include "compiler/ir/ConstModel.hpp"
#include "denox/memory/hypergraph/ConstGraph.hpp"
#include "compiler/impl/ComputeOpImpl.hpp"

namespace denox::compiler::diag {

[[noreturn]] void failed_to_realize(
    const OpModel &opModel,
    const memory::ConstGraph<TensorInstance,
                             denox::compiler::impl::details::ComputeOpImpl,
                             float> &supergraph);
}
