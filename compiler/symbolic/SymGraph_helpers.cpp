#include "symbolic/SymGraph.hpp"

namespace denox::compiler {

std::pair<SymGraph::value_type, SymGraph::value_type>
SymGraph::floordivmod(value_type a, value_type b) {
  assert(b > 0);
  value_type q = a / b;
  value_type r = a % b;
  if (r < 0) {
    --q;
    r += b;
  }
  return {q, r};
}

bool SymGraph::includes_multiset(std::span<const Sym> A,
                                 std::span<const Sym> B) {
  std::size_t i = 0, j = 0;
  while (i < A.size() && j < B.size()) {
    if (A[i].sym() < B[j].sym()) {
      ++i;
    } else if (B[j].sym() < A[i].sym()) {
      return false;
    } else {
      ++i;
      ++j;
    }
  }
  return j == B.size();
}
denox::memory::small_vector<Sym, 2>
SymGraph::multiset_diff(std::span<const Sym> A,
                        std::span<const Sym> B) { // can be const too
  denox::memory::small_vector<Sym, 2> out;
  std::size_t i = 0, j = 0;
  while (i < A.size() && j < B.size()) {
    if (A[i].sym() < B[j].sym()) {
      out.push_back(A[i++]);
    } else if (B[j].sym() < A[i].sym()) {
      ++j;
    } else {
      ++i;
      ++j;
    }
  }
  for (; i < A.size(); ++i)
    out.push_back(A[i]);
  return out;
}

SymGraph::value_type SymGraph::emod(value_type lhs, value_type rhs) {
  assert(rhs > 0);
  value_type mod = lhs % rhs;
  if (mod < 0) {
    mod += rhs;
  }
  return mod;
}
void SymGraph::emod_affine(AffineExpr &expr, value_type m) {
  auto it = expr.coef.begin();
  while (it != expr.coef.end()) {
    value_type factor = emod(it->factor, m);
    if (factor == 0) {
      it = expr.coef.erase(it);
      continue;
    }
    it->factor = factor;
    ++it;
  }
  expr.constant = emod(expr.constant, m);
}

} // namespace vkcnn
