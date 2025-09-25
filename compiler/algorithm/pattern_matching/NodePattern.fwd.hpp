#pragma once

#include "memory/container/shared_ptr.hpp"
#include "memory/hypergraph/NullWeight.hpp"
namespace denox::algorithm {

template <typename V, typename E, typename W>
class NodePattern;

template <typename V, typename E, typename W = memory::NullWeight>
using NodePatternHandle = memory::shared_ptr<NodePattern<V, E, W>>;

}
