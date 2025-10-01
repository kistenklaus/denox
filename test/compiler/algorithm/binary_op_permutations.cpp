#include "algorithm/binary_op_permutations.hpp"
#include <gtest/gtest.h>
#include <unordered_set>

using namespace denox;
using namespace algorithm;

// --- Helpers ---------------------------------------------------------------

// Expected count = prod_{m=2..n} C(m,2) = (n! * (n-1)!) / 2^(n-1).
// We compute iteratively with 128-bit intermediates to avoid overflow for small
// n.
static std::uint64_t expected_count(std::uint32_t n) {
  if (n < 2)
    return 0;
  __uint128_t acc = 1;
  for (std::uint32_t m = 2; m <= n; ++m) {
    acc *= (static_cast<__uint128_t>(m) * (m - 1)) / 2;
  }
  return static_cast<std::uint64_t>(acc);
}

// Validate structure of a single permutation for n variables.
static void validate_permutation(const BinaryOpPermutation &p,
                                 std::uint32_t n) {
  const auto &ops = p.ops;
  // 1) Exactly n-1 operations
  ASSERT_EQ(ops.size(), n == 0 ? 0u : (n < 2 ? 0u : n - 1));

  // Track variable consumption and intermediate usages
  std::vector<bool> var_used(n, false);
  std::vector<std::uint32_t> inter_used(ops.size(), 0);

  for (std::size_t i = 0; i < ops.size(); ++i) {
    const BinaryOp &op = ops[i];

    // lhs
    if (op.lhsIntermediate) {
      ASSERT_LT(op.lhs, i) << "LHS intermediate must refer to earlier op";
      inter_used[op.lhs] += 1;
    } else {
      ASSERT_LT(op.lhs, n) << "LHS variable index out of range";
      ASSERT_FALSE(var_used[op.lhs]) << "Variable used more than once";
      var_used[op.lhs] = true;
    }

    // rhs
    if (op.rhsIntermediate) {
      ASSERT_LT(op.rhs, i) << "RHS intermediate must refer to earlier op";
      inter_used[op.rhs] += 1;
    } else {
      ASSERT_LT(op.rhs, n) << "RHS variable index out of range";
      ASSERT_FALSE(var_used[op.rhs]) << "Variable used more than once";
      var_used[op.rhs] = true;
    }
  }

  // 2) Every variable must be consumed exactly once
  for (std::uint32_t v = 0; v < n; ++v) {
    ASSERT_TRUE(var_used[v]) << "Variable " << v << " was not used";
  }

  // 3) Every intermediate except the last is used exactly once
  if (!ops.empty()) {
    for (std::size_t i = 0; i + 1 < ops.size(); ++i) {
      ASSERT_EQ(inter_used[i], 1u)
          << "Intermediate " << i << " must be used exactly once";
    }
    // 4) The final result (last op) is not used by any later op
    ASSERT_EQ(inter_used.back(), 0u) << "Final result must not be used again";
  }
}

// --- Tests -----------------------------------------------------------------

TEST(algorithm_binary_op_permutations, CountsForSmallN) {
  // n = 0,1 produce no permutations
  {
    auto perms = binary_op_permutation(0);
    EXPECT_EQ(perms.size(), 0u);
  }
  {
    auto perms = binary_op_permutation(1);
    EXPECT_EQ(perms.size(), 0u);
  }

  // Verify counts for n = 2..7 (avoid n>=8 to keep runtime and memory sane)
  for (std::uint32_t n = 2; n <= 7; ++n) {
    auto perms = binary_op_permutation(n);
    const std::uint64_t want = expected_count(n);
    EXPECT_EQ(perms.size(), want) << "Unexpected count for n=" << n;
  }
}

TEST(algorithm_binary_op_permutations, StructuralValidityAllPerms_n5) {
  const std::uint32_t n = 5;
  auto perms = binary_op_permutation(n);

  // Sanity: expected count for n=5 is 180
  EXPECT_EQ(perms.size(), expected_count(n));

  for (const auto &p : perms) {
    validate_permutation(p, n);
  }
}

// Optional: a slightly larger structural pass (still reasonable runtime)
TEST(algorithm_binary_op_permutations, StructuralValidityAllPerms_n6) {
  const std::uint32_t n = 6;
  auto perms = binary_op_permutation(n);

  // Expected: 2700
  EXPECT_EQ(perms.size(), expected_count(n));

  for (const auto &p : perms) {
    validate_permutation(p, n);
  }
}
// Helper: encode a permutation into a stable string for hashing
static std::string encode_perm(const BinaryOpPermutation& p) {
  std::string s;
  s.reserve(p.ops.size() * 16);
  for (const auto& op : p.ops) {
    s.push_back(op.lhsIntermediate ? 'I' : 'V');
    s.push_back(':');
    s += std::to_string(op.lhs);
    s.push_back('|');
    s.push_back(op.rhsIntermediate ? 'I' : 'V');
    s.push_back(':');
    s += std::to_string(op.rhs);
    s.push_back(';');
  }
  return s;
}

// Helper: compute min-leaf index for each op, iteratively
static void compute_minleaf(const BinaryOpPermutation& p,
                            [[maybe_unused]] std::uint32_t n,
                            std::vector<std::uint32_t>& out_minleaf) {
  out_minleaf.resize(p.ops.size());
  for (std::size_t i = 0; i < p.ops.size(); ++i) {
    const auto& op = p.ops[i];
    std::uint32_t lhsMin = op.lhsIntermediate ? out_minleaf[op.lhs] : op.lhs;
    std::uint32_t rhsMin = op.rhsIntermediate ? out_minleaf[op.rhs] : op.rhs;
    out_minleaf[i] = std::min(lhsMin, rhsMin);
  }
}

// 1) Duplicates: none (set size equals vector size)
TEST(algorithm_binary_op_permutations, NoDuplicatePermutations_n6) {
  const std::uint32_t n = 6; // ~2700 perms, fast
  auto perms = binary_op_permutation(n);

  std::unordered_set<std::string> set;
  set.reserve(perms.size() * 2);

  for (const auto& p : perms) {
    set.insert(encode_perm(p));
  }
  EXPECT_EQ(set.size(), perms.size());
}

// 2) Canonical LHS: min-leaf(lhs) <= min-leaf(rhs) for every op
TEST(algorithm_binary_op_permutations, CanonicalLHS_n6) {
  const std::uint32_t n = 6;
  auto perms = binary_op_permutation(n);

  std::vector<std::uint32_t> minleaf;
  for (const auto& p : perms) {
    compute_minleaf(p, n, minleaf);
    for (std::size_t i = 0; i < p.ops.size(); ++i) {
      const auto& op = p.ops[i];
      std::uint32_t lhsMin = op.lhsIntermediate ? minleaf[op.lhs] : op.lhs;
      std::uint32_t rhsMin = op.rhsIntermediate ? minleaf[op.rhs] : op.rhs;
      EXPECT_LE(lhsMin, rhsMin);
    }
  }
}

// 3) Determinism: two runs produce exactly the same multiset
TEST(algorithm_binary_op_permutations, Determinism_n6) {
  const std::uint32_t n = 6;
  auto perms1 = binary_op_permutation(n);
  auto perms2 = binary_op_permutation(n);

  std::unordered_multiset<std::string> s1, s2;
  s1.reserve(perms1.size()); s2.reserve(perms2.size());
  for (const auto& p : perms1) s1.insert(encode_perm(p));
  for (const auto& p : perms2) s2.insert(encode_perm(p));
  EXPECT_EQ(s1.size(), s2.size());
  EXPECT_EQ(s1, s2);
}

// 4) Semantic sanity: evaluating with + yields same sum for all permutations
TEST(algorithm_binary_op_permutations, SemanticSanityPlus_n5) {
  const std::uint32_t n = 5;
  auto perms = binary_op_permutation(n);

  // Assign arbitrary values 0..n-1
  std::vector<int> vals(n);
  for (std::uint32_t i = 0; i < n; ++i) vals[i] = static_cast<int>(i);

  const int baseline = (n - 1) * n / 2; // sum 0..n-1

  for (const auto& p : perms) {
    std::vector<int> opval(p.ops.size(), 0);
    for (std::size_t i = 0; i < p.ops.size(); ++i) {
      const auto& op = p.ops[i];
      int L = op.lhsIntermediate ? opval[op.lhs] : vals[op.lhs];
      int R = op.rhsIntermediate ? opval[op.rhs] : vals[op.rhs];
      opval[i] = L + R;
    }
    ASSERT_EQ(opval.empty() ? 0 : opval.back(), baseline);
  }
}
