#pragma once

#include "memory/container/shared_ptr.hpp"
namespace denox::algorithm {

template <typename V, typename E, typename W>
class NodePattern;

template <typename V, typename E, typename W>
using NodePatternHandle = memory::shared_ptr<NodePattern<V, E, W>>;

}
