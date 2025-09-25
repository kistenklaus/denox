#pragma once

#include "memory/container/shared_ptr.hpp"
#include "memory/hypergraph/NullWeight.hpp"
namespace denox::algorithm {

template <typename V, typename E, typename W> class EdgePattern;

template <typename V, typename E, typename W = memory::NullWeight>
using EdgePatternHandle = memory::shared_ptr<EdgePattern<V, E, W>>;

} // namespace denox::algorithm
