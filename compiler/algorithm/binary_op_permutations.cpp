#include "algorithm/binary_op_permutations.hpp"
#include <cassert>

namespace denox::algorithm {

struct Bucket {
  // If false → 'ref' is a variable index [0..n-1].
  // If true  → 'ref' is an op index into 'ops' (the result of that op).
  bool isIntermediate;
  std::uint32_t ref;

  // Minimal original variable index contained in this bucket.
  // Used to canonicalize operand ordering and avoid (A,B) vs (B,A) duplicates.
  std::uint32_t minLeaf;
};

static void
enumerate_permutations_rec(const memory::vector<Bucket> &buckets,
                           memory::vector<BinaryOp> &ops,
                           memory::vector<BinaryOpPermutation> &out) {
  const std::size_t m = buckets.size();
  if (m == 1) {
    // One bucket left → a complete reduction.
    BinaryOpPermutation p;
    p.ops = ops;
    out.push_back(std::move(p));
    return;
  }

  // Pick every unordered pair of buckets (i < j), combine them, recurse.
  for (std::size_t i = 0; i + 1 < m; ++i) {
    for (std::size_t j = i + 1; j < m; ++j) {
      const Bucket &bi = buckets[i];
      const Bucket &bj = buckets[j];

      // Canonicalize operand order to avoid (A,B) vs (B,A).
      const Bucket *left = &bi;
      const Bucket *right = &bj;
      if (right->minLeaf < left->minLeaf) {
        left = &bj;
        right = &bi;
      }

      // Emit the op combining these two buckets.
      BinaryOp op;
      op.lhsIntermediate = left->isIntermediate;
      op.lhs = left->ref;
      op.rhsIntermediate = right->isIntermediate;
      op.rhs = right->ref;

      ops.push_back(op);
      const std::uint32_t newOpIndex =
          static_cast<std::uint32_t>(ops.size() - 1);

      // Build next bucket list: all except i and j, plus the new merged bucket.
      memory::vector<Bucket> next;
      next.reserve(m - 1);
      for (std::size_t t = 0; t < m; ++t) {
        if (t == i || t == j)
          continue;
        next.push_back(buckets[t]);
      }
      Bucket merged;
      merged.isIntermediate = true;
      merged.ref = newOpIndex;
      merged.minLeaf = std::min(left->minLeaf, right->minLeaf);
      next.push_back(merged);

      // Recurse
      enumerate_permutations_rec(next, ops, out);

      // Backtrack
      ops.pop_back();
    }
  }
}

memory::vector<BinaryOpPermutation> binary_op_permutation(std::uint32_t n) {
  memory::vector<BinaryOpPermutation> out;
  if (n < 2)
    return out;

  // Initial buckets: one per variable
  memory::vector<Bucket> buckets;
  buckets.reserve(n);
  for (std::uint32_t v = 0; v < n; ++v) {
    Bucket b;
    b.isIntermediate = false;
    b.ref = v;     // variable index
    b.minLeaf = v; // itself
    buckets.push_back(b);
  }

  memory::vector<BinaryOp> ops;
  ops.reserve(n - 1);

  enumerate_permutations_rec(buckets, ops, out);
  return out;
}

} // namespace denox::algorithm
