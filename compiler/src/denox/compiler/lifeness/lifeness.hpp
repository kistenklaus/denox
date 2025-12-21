#pragma once

#include "denox/compiler/canonicalize/CanoModel.hpp"
#include "denox/compiler/lifeness/Lifetimes.hpp"
namespace denox::compiler {

/**
 * Computes value lifetimes for nodes in a CanoModel computation graph.
 *
 * Each node is assigned a half-open lifetime interval [start, end),
 * expressed in abstract execution time units derived from a topological
 * schedule of the graph.
 *
 * Semantics:
 *  - start: earliest execution time at which the node's value is produced
 *  - end:   latest time any consumer may still require the value (exclusive)
 *
 * Node classification:
 *  - active   : reachable from inputs AND can reach outputs
 *  - present  : reachable from inputs OR can reach outputs
 *
 * Lifetime conventions:
 *  - active nodes        -> meaningful [start, end) interval
 *  - present but inactive-> {0, 0}
 *  - not present         -> {UINT64_MAX, UINT64_MAX}
 *
 * The result is suitable for memory planning, buffer reuse, and
 * liveness-aware scheduling passes.
 */
Lifetimes lifeness(const CanoModel &model);

} // namespace denox::compiler
