#include "./SymGraph.hpp"


namespace vkcnn {

const SymGraph::ModSolverHandle &SymGraph::require_modsolver(Sym sym) {
  auto it = m_modSolverCache.find(sym);
  if (it == m_modSolverCache.end()) {
    auto solver = std::make_shared<symbolic::details::ModSolver>();
    return m_modSolverCache.insert(it, std::make_pair(sym, std::move(solver)))
        ->second;
  } else {
    return it->second;
  }
}

}
