#pragma once

#include "vkcnn/common/symbolic/SymVar.hpp"
#include <cstdint>
#include <vector>
namespace vkcnn {

class SymEval {
public:
  std::uint64_t operator[](hypergraph::NodeId id) const { return m_eval[id]; }

  SymEval(std::vector<std::uint64_t> eval) : m_eval(std::move(eval)) {}

private:
  std::vector<std::uint64_t> m_eval;
};

} // namespace vkcnn
