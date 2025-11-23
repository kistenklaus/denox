#include "symbolic/SymGraph.hpp"
#include <gtest/gtest.h>

using namespace denox::compiler;

#define NF(g, e) (g).resolve((e))
#define EQ_NF(g, a, b) EXPECT_EQ(NF(g, a), NF(g, b))

#define EXPECT_SEMANTIC_EQ(G, LHS, RHS)                                        \
  do {                                                                         \
    auto __d = (G).resolve((G).sub((LHS), (RHS)));                             \
    ASSERT_FALSE(__d.isSymbolic());                                            \
    EXPECT_EQ(__d.constant(), 0);                                              \
  } while (0)

// Commutativity: A*B == B*A  (should already pass)
TEST(symbolic, nonaffine_mul_commutative_semantic) {
  SymGraph g;
  auto A = g.var(), B = g.var();
  EXPECT_SEMANTIC_EQ(g, g.mul(A, B), g.mul(B, A));
}

// Associativity: (A*B)*C == A*(B*C)  (semantic truth; may fail until you
// canonicalize products)
TEST(symbolic, nonaffine_mul_associative_semantic) {
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var();
  EXPECT_SEMANTIC_EQ(g, g.mul(g.mul(A, B), C), g.mul(A, g.mul(B, C)));
}

// Left distributivity: A*(B+C) == A*B + A*C
TEST(symbolic, nonaffine_left_distributive_semantic) {
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var();
  EXPECT_SEMANTIC_EQ(g, g.mul(A, g.add(B, C)), g.add(g.mul(A, B), g.mul(A, C)));
}

// Right distributivity: (A+B)*C == A*C + B*C
TEST(symbolic, nonaffine_right_distributive_semantic) {
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var();
  EXPECT_SEMANTIC_EQ(g, g.mul(g.add(A, B), C), g.add(g.mul(A, C), g.mul(B, C)));
}

// Scalar commutation & associativity with product: k*(A*B) == (k*A)*B ==
// A*(k*B)
TEST(symbolic, nonaffine_scalar_product_semantic) {
  SymGraph g;
  auto A = g.var(), B = g.var();
  EXPECT_SEMANTIC_EQ(g, g.mul(3, g.mul(A, B)), g.mul(g.mul(3, A), B));
  EXPECT_SEMANTIC_EQ(g, g.mul(3, g.mul(A, B)), g.mul(A, g.mul(3, B)));
}

// Neutral/absorbing elements: 1*E == E, E*1 == E, 0*E == 0, E*0 == 0
TEST(symbolic, nonaffine_neutral_absorbing_semantic) {
  SymGraph g;
  auto A = g.var(), B = g.var();
  auto E = g.add(A, B); // any positive expression
  EXPECT_SEMANTIC_EQ(g, g.mul(1, E), E);
  EXPECT_SEMANTIC_EQ(g, g.mul(E, 1), E);
  auto z1 = g.resolve(g.mul(0, E));
  ASSERT_FALSE(z1.isSymbolic());
  EXPECT_EQ(z1.constant(), 0);
  auto z2 = g.resolve(g.mul(E, 0));
  ASSERT_FALSE(z2.isSymbolic());
  EXPECT_EQ(z2.constant(), 0);
}

// Same-base scaled mod: (k·E) % (m·E) == (k%m)·E
TEST(symbolic, nonaffine_mod_same_base_scaled_semantic) {
  SymGraph g;
  auto A = g.var(), B = g.var();
  auto E = g.add(g.mul(A, B), 1); // ensure E > 0
  EXPECT_SEMANTIC_EQ(g, g.mod(g.mul(5, E), g.mul(3, E)), g.mul(5 % 3, E)); // 2E
  auto r0 = g.resolve(g.mod(g.mul(6, E), g.mul(3, E)));
  ASSERT_FALSE(r0.isSymbolic());
  EXPECT_EQ(r0.constant(), 0);
}

// Mod by self: E % E == 0
TEST(symbolic, nonaffine_mod_self_zero_semantic) {
  SymGraph g;
  auto A = g.var(), B = g.var();
  auto E = g.add(g.mul(A, B), g.add(A, 2));
  auto r = g.resolve(g.mod(E, E));
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(r.constant(), 0);
}

// Mod by a factor: (A*B) % A == 0
TEST(symbolic, nonaffine_mod_by_factor_zero_semantic) {
  SymGraph g;
  auto A = g.var(), B = g.var();
  auto r = g.resolve(g.mod(g.mul(A, B), A));
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(r.constant(), 0);
}

// Exact cancel in division (requires positivity; semantic truth)
// May fail until you implement factor-cancel in div.
TEST(symbolic, nonaffine_exact_div_cancel_semantic) {
  SymGraph g;
  auto A = g.var(), B = g.var();
  EXPECT_SEMANTIC_EQ(g, g.div(g.mul(A, B), A), B);
}

// Scaled exact cancel: (k·E)/k == E (k>0)
TEST(symbolic, nonaffine_scaled_exact_div_cancel_semantic) {
  SymGraph g;
  auto A = g.var();
  auto E = g.add(g.mul(4, A), 6);
  EXPECT_SEMANTIC_EQ(g, g.div(g.mul(5, E), 5), E);
}

// GCD-scaled div same base: (k·E)/(m·E) == floor(k/m)
TEST(symbolic, nonaffine_gcd_scaled_div_same_base_semantic) {
  SymGraph g;
  auto A = g.var(), B = g.var();
  auto E = g.add(g.mul(A, B), 1);
  auto q = g.resolve(g.div(g.mul(7, E), g.mul(3, E))); // floor(7/3)=2
  ASSERT_FALSE(q.isSymbolic());
  EXPECT_EQ(q.constant(), 2);
}

// A*(B + C) == A*B + A*C
TEST(symbolic, nonaffine_left_distributivity) {
  SymGraph g;
  auto A = g.var();
  auto B = g.var();
  auto C = g.var();

  auto lhs = g.mul(A, g.add(B, C));
  auto rhs = g.add(g.mul(A, B), g.mul(A, C));
  EXPECT_EQ(lhs, rhs);
}

// Same as above but with (C + B) to catch any order-sensitivity bugs
TEST(symbolic, nonaffine_left_distributivity_comm_order) {
  SymGraph g;
  auto A = g.var();
  auto B = g.var();
  auto C = g.var();

  auto lhs = g.mul(A, g.add(C, B)); // flipped order inside sum
  auto rhs = g.add(g.mul(A, B), g.mul(A, C));
  EXPECT_EQ(lhs, rhs);
}

// (A + B)*C == A*C + B*C  (right-distributivity)
TEST(symbolic, nonaffine_right_distributivity) {
  SymGraph g;
  auto A = g.var();
  auto B = g.var();
  auto C = g.var();

  auto lhs = g.mul(g.add(A, B), C);
  auto rhs = g.add(g.mul(A, C), g.mul(B, C));
  EXPECT_EQ(lhs, rhs);
}

// A*(B + C) == A*B + A*C  (left distributivity – variant 2)
TEST(symbolic, nonaffine_left_distributivity_nested) {
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var(), D = g.var();
  auto lhs = g.mul(A, g.add(B, g.add(C, D)));
  auto rhs = g.add(g.add(g.mul(A, B), g.mul(A, C)), g.mul(A, D));
  auto d = g.resolve(g.sub(lhs, rhs));
  ASSERT_FALSE(d.isSymbolic());
  EXPECT_EQ(d.constant(), 0);
}

// (A + B)*C == A*C + B*C  (right distributivity – variant 2)
TEST(symbolic, nonaffine_right_distributivity_nested) {
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var(), D = g.var();
  auto lhs = g.mul(g.add(A, g.add(B, C)), D);
  auto rhs = g.add(g.add(g.mul(A, D), g.mul(B, D)), g.mul(C, D));
  EXPECT_EQ(lhs, rhs);
}

// Full bilinear expansion: (A + B) * (C + D) == AC + AD + BC + BD
TEST(symbolic, nonaffine_bilinear_expansion) {
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var(), D = g.var();
  auto lhs = g.mul(g.add(A, B), g.add(C, D));
  auto rhs =
      g.add(g.add(g.mul(A, C), g.mul(A, D)), g.add(g.mul(B, C), g.mul(B, D)));
  EXPECT_EQ(lhs, rhs);
}

// Distribute with constants inside sum: A*(B + 2) == A*B + 2*A
TEST(symbolic, nonaffine_left_distributivity_with_const) {
  SymGraph g;
  auto A = g.var(), B = g.var();
  auto lhs = g.mul(A, g.add(B, 2));
  auto rhs = g.add(g.mul(A, B), g.mul(2, A));
  EXPECT_EQ(lhs, rhs);
}

// Distribute with constants (right): (A + 2)*B == A*B + 2*B
TEST(symbolic, nonaffine_right_distributivity_with_const) {
  SymGraph g;
  auto A = g.var(), B = g.var();
  auto lhs = g.mul(g.add(A, 2), B);
  auto rhs = g.add(g.mul(A, B), g.mul(2, B));
  EXPECT_EQ(lhs, rhs);
}

// Factor common product: A*B + A*C == A*(B + C)
TEST(symbolic, nonaffine_factor_common_left) {
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var();
  auto lhs = g.add(g.mul(A, B), g.mul(A, C));
  auto rhs = g.mul(A, g.add(B, C));
  EXPECT_EQ(lhs, rhs);
}

// Factor common product with scalar: 2*A*B + 2*A*C == 2*A*(B + C)
TEST(symbolic, nonaffine_factor_common_with_scalar) {
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var();
  auto lhs = g.add(g.mul(g.mul(2, A), B), g.mul(g.mul(2, A), C));
  auto rhs = g.mul(g.mul(2, A), g.add(B, C));
  EXPECT_EQ(lhs, rhs);
}

// Combine distributivity across three terms: A*(B + C) + A*D == A*(B + C + D)
TEST(symbolic, nonaffine_distribute_three_terms) {
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var(), D = g.var();
  auto lhs = g.add(g.mul(A, g.add(B, C)), g.mul(A, D));
  auto rhs = g.mul(A, g.add(g.add(B, C), D));
  EXPECT_EQ(lhs, rhs);
}

// Pull scalar through sum of products: k*(A*B + A*C) == A*(k*B + k*C)
TEST(symbolic, nonaffine_scalar_through_sum_of_products) {
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var();
  auto k = 3;
  auto lhs = g.mul(k, g.add(g.mul(A, B), g.mul(A, C)));
  auto rhs = g.mul(A, g.add(g.mul(k, B), g.mul(k, C)));
  EXPECT_EQ(lhs, rhs);
}

// Mod with same base wrapped in sums/products: (k·E)%(m·E) = (k%m)·E, with E
// nontrivial
TEST(symbolic, nonaffine_mod_same_base_scaled_complex_E) {
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var();
  auto E = g.mul(g.add(A, 1), g.add(B, C)); // E = (A+1)*(B+C)
  auto lhs = g.mod(g.mul(7, E), g.mul(3, E));
  auto rhs = g.mul(7 % 3, E); // 2*E
  EXPECT_EQ(lhs, rhs);
}

// A*(B + C) == A*B + A*C     (left distributivity)  — likely FAIL until you
// distribute into sums
TEST(symbolic, nonaffine_left_distributivity_structural) {
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var();

  auto lhs = g.mul(A, g.add(B, C));
  auto rhs = g.add(g.mul(A, B), g.mul(A, C));
  EXPECT_EQ(lhs, rhs);
}

// (A + B)*C == A*C + B*C      (right distributivity) — you said this currently
// FAILs
TEST(symbolic, nonaffine_right_distributivity_structural) {
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var();

  auto lhs = g.mul(g.add(A, B), C);
  auto rhs = g.add(g.mul(A, C), g.mul(B, C));
  EXPECT_EQ(lhs, rhs);
}

// (A + B)*(C + D) == AC + AD + BC + BD  — full bilinear expansion
TEST(symbolic, nonaffine_bilinear_expansion_structural) {
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var(), D = g.var();

  auto lhs = g.mul(g.add(A, B), g.add(C, D));
  auto rhs =
      g.add(g.add(g.mul(A, C), g.mul(A, D)), g.add(g.mul(B, C), g.mul(B, D)));
  EXPECT_EQ(lhs, rhs);
}

// Factor out common left factor: A*B + A*C == A*(B + C)
TEST(symbolic, nonaffine_factor_common_left_structural) {
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var();

  auto lhs = g.add(g.mul(A, B), g.mul(A, C));
  auto rhs = g.mul(A, g.add(B, C));
  EXPECT_EQ(lhs, rhs);
}

// Factor out common right factor: B*A + C*A == (B + C)*A
TEST(symbolic, nonaffine_factor_common_right_structural) {
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var();

  auto lhs = g.add(g.mul(B, A), g.mul(C, A));
  auto rhs = g.mul(g.add(B, C), A);
  EXPECT_EQ(lhs, rhs);
}

// Merge identical product terms: (AB + BA) == 2*(AB)
TEST(symbolic, nonaffine_merge_identical_products_structural) {
  SymGraph g;
  auto A = g.var(), B = g.var();

  auto AB = g.mul(A, B);
  auto BA = g.mul(B, A);
  auto lhs = g.add(AB, BA);
  auto rhs = g.mul(2, AB);
  EXPECT_EQ(lhs, rhs);
}

// Scalar movement through product: 3*(A*B) == (3*A)*B == A*(3*B)  — should PASS
// if you commute scalars
TEST(symbolic, nonaffine_scalar_move_through_product_structural) {
  SymGraph g;
  auto A = g.var(), B = g.var();

  auto left = g.mul(3, g.mul(A, B));
  auto mid = g.mul(g.mul(3, A), B);
  auto right = g.mul(A, g.mul(3, B));
  EXPECT_EQ(left, mid);
  EXPECT_EQ(left, right);
}

// Exact cancel in division: (A*(B+C))/A == (B+C)  — likely FAIL until you
// implement factor-cancel
TEST(symbolic, nonaffine_div_cancel_common_factor_sum_structural) {
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var();

  auto lhs = g.div(g.mul(A, g.add(B, C)), A);
  auto rhs = g.add(B, C);
  EXPECT_EQ(lhs, rhs);
}

// Scaled exact cancel: (6*(B+C))/6 == B+C
TEST(symbolic, nonaffine_div_cancel_scalar_sum_structural) {
  SymGraph g;
  auto B = g.var(), C = g.var();

  auto lhs = g.div(g.mul(6, g.add(B, C)), 6);
  auto rhs = g.add(B, C);
  EXPECT_EQ(lhs, rhs);
}

// GCD-scaled div with same base: (k·E)/(m·E) == floor(k/m)
// Compare against an explicit constant: build 2 as div(4,2)
TEST(symbolic, nonaffine_div_same_base_scale_structural) {
  SymGraph g;
  auto A = g.var(), B = g.var();
  auto E = g.add(g.mul(A, B), 1); // E > 0

  auto lhs = g.div(g.mul(7, E), g.mul(3, E)); // expect 2
  auto rhs = g.div(4, 2);                     // constant-2 node
  EXPECT_EQ(lhs, rhs);
}

// Mod same base scaled: (k·E) % (m·E) == (k%m)·E
TEST(symbolic, nonaffine_mod_same_base_scaled_structural) {
  SymGraph g;
  auto A = g.var(), B = g.var();
  auto E = g.add(g.mul(A, B), 1);

  auto lhs = g.mod(g.mul(5, E), g.mul(3, E)); // expect 2*E
  auto rhs = g.mul(2, E);
  EXPECT_EQ(lhs, rhs);

  auto lhs0 = g.mod(g.mul(6, E), g.mul(3, E)); // expect 0
  auto rhs0 = g.mul(0, E);                     // canonical zero node
  EXPECT_EQ(lhs0, rhs0);
}

// Mod by factor: (A*B) % A == 0  — you observed FAIL now; keep as target
TEST(symbolic, nonaffine_mod_by_factor_zero_structural) {
  SymGraph g;
  auto A = g.var(), B = g.var();
  auto lhs = g.mod(g.mul(A, B), A);
  auto rhs = g.mul(0, A); // zero node
  EXPECT_EQ(lhs, rhs);
}

// (A*B + A - 1) / A == B
TEST(symbolic, nonaffine_ceildiv_one_more_block) {
  SymGraph g;
  auto a = g.var(), b = g.var();
  auto ceildiv = g.div(g.add(g.mul(a, b), g.sub(a, 1)), a);
  auto expect = b;
  EXPECT_EQ(ceildiv, expect);
}

TEST(symbolic, nonaffine_floordiv_coefficients_complex) {
  SymGraph g;
  auto A = g.var();
  auto B = g.var();

  auto lhs = g.div(g.add(g.add(g.mul(4, A), g.mul(3, B)), 3), 2);
  auto rhs = g.add(g.mul(2, A), g.div(g.add(g.mul(3, B), 3), 2));
  EXPECT_EQ(rhs, lhs);
}

TEST(symbolic, nonaffine_div_exact_no_residual) {
  SymGraph g;
  auto A = g.var();
  auto B = g.var();

  auto lhs =
      g.div(g.add(g.add(g.mul(6, A), g.mul(10, B)), 8), 2); // (6A + 10B + 8)/2
  auto rhs = g.add(g.add(g.mul(3, A), g.mul(5, B)), 4);     // 3A + 5B + 4
  EXPECT_EQ(rhs, lhs);
}

TEST(symbolic, nonaffine_div_constant_remainder_only) {
  SymGraph g;
  auto A = g.var();
  auto B = g.var();

  auto lhs =
      g.div(g.add(g.add(g.mul(6, A), g.mul(10, B)), 7), 2); // (6A + 10B + 7)/2
  auto rhs = g.add(g.add(g.mul(3, A), g.mul(5, B)), 3);     // 3A + 5B + 3
  EXPECT_EQ(rhs, lhs);
}

TEST(symbolic, nonaffine_div_mixed_normalizes_equal) {
  SymGraph g;
  auto A = g.var();
  auto B = g.var();

  auto t1 = g.div(g.add(g.add(g.mul(4, A), g.mul(3, B)), 3), 2);
  auto t2 =
      g.add(g.mul(2, A), g.div(g.add(g.mul(3, B), 3), 2)); // 2A + div(3B+3,2)

  // Canonical: 2A + B + 1 + div(B + 1, 2)
  auto t0 = g.add(g.mul(2, A), B);
  auto can = g.add(g.add(t0, 1), g.div(g.add(B, 1), 2));

  EXPECT_EQ(can, t1);
  EXPECT_EQ(can, t2);
}

TEST(symbolic, nonaffine_div_negative_constant_euclidean) {
  SymGraph g;
  auto A = g.var();

  auto lhs = g.div(g.sub(g.mul(2, A), 1), 2);
  auto rhs = g.sub(A, 1);
  EXPECT_EQ(rhs, lhs);
}

TEST(symbolic, nonaffine_div_pure_constant_residual_dropped) {
  SymGraph g;
  auto A = g.var();
  auto B = g.var();

  auto lhs =
      g.div(g.add(g.add(g.mul(8, A), g.mul(6, B)), 4), 2); // (8A + 6B + 4)/2
  auto rhs = g.add(g.add(g.mul(4, A), g.mul(3, B)), 2);    // 4A + 3B + 2
  EXPECT_EQ(rhs, lhs);
}

TEST(symbolic, nonaffine_div_by_one_is_identity) {
  SymGraph g;
  auto A = g.var();
  auto B = g.var();

  auto prod = g.mul(A, B);   // NonAffine: Mul(A,B)
  auto lhs = g.div(prod, 1); // (A*B)/1
  EXPECT_EQ(prod, lhs);
}

TEST(symbolic, nonnested_div_constants_compose) {
  SymGraph g;
  auto A = g.var();

  auto lhs = g.div(g.div(A, 4), 2); // (A div 4) div 2
  auto rhs = g.div(A, 8);           // A div 8
  EXPECT_EQ(rhs, lhs);
}

TEST(symbolic, nonconstant_over_symbol_then_div_by_bigger_constant_is_zero) {
  SymGraph g;
  auto X = g.var();

  auto lhs = g.div(g.div(1, X), 2); // (1 div X) div 2  -> 0  (since 1 < 2)
  auto rhs = g.mul(0, X); // canonical zero (or g.add(0,0) if you prefer)
  EXPECT_EQ(rhs, lhs);
}

TEST(symbolic, nonaffine_div_k_times_w_plus_k_minus_one_over_k) {
  SymGraph g;
  auto W = g.var();

  std::size_t k = 5;
  auto lhs = g.div(g.add(g.mul(k, W), k - 1), k);
  EXPECT_EQ(W, lhs);
}

TEST(symbolic, nonaffine_div_mixed_order_insensitive) {
  SymGraph g;
  auto A = g.var();
  auto B = g.var();

  auto e1 = g.div(g.add(g.add(g.mul(4, A), g.mul(3, B)), 3), 2);
  auto e2 = g.div(g.add(g.add(g.mul(3, B), 3), g.mul(4, A)), 2);
  EXPECT_EQ(e1, e2);
}

TEST(symbolic, nonaffine_affineNumerator_div_symbolic_den_rebalance_Aminus1) {
  SymGraph g;
  auto A = g.var(), B = g.var();
  auto lhs = g.div(g.add(g.mul(A, B), g.sub(A, 1)), A);
  EXPECT_EQ(B, lhs);
}

TEST(symbolic, nonaffine_div_cancels_into_div) {
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var();
  auto lhs = g.div(g.mul(A, B), g.mul(A, C));
  auto expect = g.div(B, C);
  EXPECT_EQ(expect, lhs);
}

TEST(symbolic, nonaffine_prod_gcd_simple) {
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var();
  auto lhs = g.div(g.mul(A, B), g.mul(A, C)); // AB / AC
  auto rhs = g.div(B, C);                     // B / C
  EXPECT_EQ(rhs, lhs);
}

TEST(symbolic, nonaffine_prod_gcd_equal) {
  SymGraph g;
  auto A = g.var(), B = g.var();
  auto lhs = g.resolve(g.div(g.mul(A, B), g.mul(A, B))); // AB / AB
  ASSERT_FALSE(lhs.isSymbolic());
  ;
  EXPECT_EQ(1, lhs.constant()); // == 1 (use any canonical 1)
}

TEST(symbolic, nonaffine_prod_gcd_superset_den) {
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var();
  auto lhs = g.div(g.mul(A, B), g.mul(g.mul(A, B), C)); // AB / (AB*C)
  auto rhs = g.div(1, C);                               // 1 / C
  EXPECT_EQ(rhs, lhs);
}

TEST(symbolic, nonaffine_prod_gcd_powers) {
  SymGraph g;
  auto A = g.var(), C = g.var();
  auto num = g.mul(A, C);           // A*C
  auto den = g.mul(g.mul(A, C), C); // A*C*C
  auto lhs = g.div(num, den);       // (A*C)/(A*C*C)
  auto rhs = g.div(1, C);           // 1/(C*C)
  EXPECT_EQ(rhs, lhs);
}

TEST(symbolic, nonaffine_prod_gcd_reduce_to_symbol) {
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var();
  auto lhs = g.div(g.mul(g.mul(A, B), C), g.mul(A, B)); // ABC / AB
  EXPECT_EQ(C, lhs);
}

// -------------------- nonaffine: nested div composition --------------------

TEST(symbolic, nonaffine_nested_div_symbolic) {
  SymGraph g;
  auto X = g.var(), A = g.var(), B = g.var();
  auto lhs = g.div(g.div(X, A), B); // (X/A)/B
  auto rhs = g.div(X, g.mul(A, B)); // X/(A*B)
  EXPECT_EQ(rhs, lhs);
}

TEST(symbolic, nonaffine_nested_div_symbolic_and_const) {
  SymGraph g;
  auto X = g.var(), A = g.var();
  auto lhs = g.div(g.div(X, A), 2); // (X/A)/2
  auto rhs = g.div(X, g.mul(A, 2)); // X/(A*2)
  EXPECT_EQ(rhs, lhs);
}

TEST(symbolic, nonaffine_nested_div_constants) {
  SymGraph g;
  auto X = g.var();
  auto lhs = g.div(g.div(X, 4), 2); // (X div 4) div 2
  auto rhs = g.div(X, 8);           // X div 8
  EXPECT_EQ(rhs, lhs);
}

// -------------------- nonaffine: affine numerator ÷ symbolic denom
// --------------------

TEST(symbolic, nonaffine_affine_over_symbolic_rebalance_Aminus1) {
  SymGraph g;
  auto A = g.var(), B = g.var();
  auto lhs = g.div(g.add(g.mul(A, B), g.sub(A, 1)), A); // (AB + A - 1)/A
  EXPECT_EQ(B, lhs);
}

TEST(symbolic, nonaffine_affine_over_composite_rebalance) {
  SymGraph g;
  auto A = g.var(), B = g.var(), X = g.var();
  auto D = g.mul(g.mul(A, B), X);                        // D = A*B*X
  auto lhs = g.resolve(g.div(g.add(D, g.sub(D, 1)), D)); // (D + D - 1)/D
  ASSERT_FALSE(lhs.isSymbolic());
  EXPECT_EQ(1, lhs.constant()); // == 1
}

TEST(symbolic, nonaffine_termwise_cancellation_with_residual_div) {
  SymGraph g;
  auto A = g.var(), X = g.var(), Y = g.var(), Z = g.var();
  auto lhs =
      g.div(g.add(g.add(g.mul(g.mul(2, A), X), g.mul(g.mul(3, A), Y)), Z), A);
  // Expect: 2X + 3Y + div(Z, A)
  auto rhs = g.add(g.add(g.mul(2, X), g.mul(3, Y)), g.div(Z, A));
  EXPECT_EQ(rhs, lhs);
}

TEST(symbolic, nonaffine_residual_constant_one_kept) {
  SymGraph g;
  auto A = g.var(), B = g.var();
  auto lhs = g.div(g.add(g.add(g.mul(A, B), A), 1), A); // (AB + A + 1)/A
  auto rhs = g.add(g.add(B, 1), g.div(1, A));           // B + 1 + (1/A)
  EXPECT_EQ(rhs, lhs);
}

// ---------- simple wins

TEST(symbolic, nonaffine_mod_const_absorption_gcd) {
  SymGraph g;
  auto X = g.var();
  auto lhs = g.mod(g.mod(X, 6), 4); // (X % 6) % 4
  auto rhs = g.mod(X, 2);           // X % gcd(6,4) = X % 2
  EXPECT_EQ(rhs, lhs);
}

TEST(symbolic, nonaffine_mod_idempotent_symbolic) {
  SymGraph g;
  auto X = g.var(), A = g.var();
  auto lhs = g.mod(g.mod(X, A), A); // (X % A) % A
  auto rhs = g.mod(X, A);           // X % A
  EXPECT_EQ(rhs, lhs);
}

TEST(symbolic, nonaffine_mod_product_divides_zero) {
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var();
  EXPECT_EQ(g.mul(0, A), g.mod(g.mul(A, B), A)); // (A*B) % A == 0
  EXPECT_EQ(g.mul(0, A),
            g.mod(g.mul(g.mul(A, B), C), g.mul(A, B))); // (A*B*C) % (A*B) == 0
}

TEST(symbolic, nonaffine_mod_same_symbol_zero) {
  SymGraph g;
  auto A = g.var();
  EXPECT_EQ(g.mul(0, A), g.mod(A, A)); // A % A == 0
}

TEST(symbolic, nonaffine_mod_trivial_constants) {
  SymGraph g;
  auto X = g.var();
  EXPECT_EQ(g.mul(0, X), g.mod(0, X)); // 0 % X == 0
  auto one = g.resolve(g.mod(X, 1));   // X % 1 == 0
  ASSERT_FALSE(one.isSymbolic());
  EXPECT_EQ(0, one.constant());
}

// ============================================================================
// PROVABLE, NONTRIVIAL EQUALITIES (should PASS)
// Prefix: nonaffine_
// ============================================================================

TEST(symbolic, nonaffine_right_distributive_semantic_deep) {
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var();
  auto lhs = g.mul(g.add(A, B), C);
  auto rhs = g.add(g.mul(A, C), g.mul(B, C));
  EXPECT_EQ(NF(g, lhs), NF(g, rhs));
}

// Product cancellation with constants folded as affine scaling:
// (A*B*C*X) / (A*B) == C*X
TEST(symbolic, nonaffine_product_cancel_with_constants) {
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var(), X = g.var();
  auto lhs = g.div(g.mul(g.mul(g.mul(A, B), C), X), g.mul(A, B));
  auto rhs = g.mul(C, X);
  EXPECT_EQ(NF(g, lhs), NF(g, rhs));
}

// Nested floor associativity (constants): floor(floor(X/a)/b) == floor(X/(a*b))
TEST(symbolic, nonaffine_nested_floor_associativity_constants) {
  SymGraph g;
  auto X = g.var();
  int a = 3, b = 4, ab = a * b;
  auto lhs = g.div(g.div(X, a), b);
  auto rhs = g.div(X, ab);
  EXPECT_EQ(NF(g, lhs), NF(g, rhs));
}

// Partial peel by single symbol (no divisibility assumption on residual):
// (B*X + R) / B == X + floor(R/B)
TEST(symbolic, nonaffine_partial_peel_single_symbol) {
  SymGraph g;
  auto B = g.var(), X = g.var(), R = g.var();
  auto lhs = g.div(g.add(g.mul(B, X), R), B);
  auto rhs = g.add(X, g.div(R, B));
  EXPECT_EQ(NF(g, lhs), NF(g, rhs));
}

// Full peel of common symbol then divide by constant:
// (B*X + B*Y + B*Z) / (k*B) == floor((X+Y+Z)/k)
TEST(symbolic, nonaffine_full_peel_then_div_by_const_sum) {
  SymGraph g;
  auto B = g.var(), X = g.var(), Y = g.var(), Z = g.var();
  int k = 5;
  auto lhs =
      g.div(g.add(g.add(g.mul(B, X), g.mul(B, Y)), g.mul(B, Z)), g.mul(k, B));
  auto rhs = g.div(g.add(g.add(X, Y), Z), k);
  EXPECT_EQ(NF(g, lhs), NF(g, rhs));
}

// Pooling on multiples (K=1,P=0,D=1): pool(S*X, S) == X
TEST(symbolic, nonaffine_pool_on_multiple_exact) {
  SymGraph g;
  auto S = g.var(), X = g.var();
  auto lhs = g.pool(g.mul(S, X), 1, 0, S, 1);
  EXPECT_EQ(NF(g, lhs), X);
}

// Two-stage pooling when perfectly factored: pool(pool(S*T*X, S), T) == X
TEST(symbolic, nonaffine_pool_two_stage_exact_product) {
  SymGraph g;
  auto S = g.var(), T = g.var(), X = g.var();
  auto lhs = g.pool(g.pool(g.mul(g.mul(S, T), X), 1, 0, S, 1), 1, 0, T, 1);
  EXPECT_EQ(NF(g, lhs), X);
}

// AlignUp idempotence for constant A: alignUp(alignUp(X,A), A) == alignUp(X,A)
TEST(symbolic, nonaffine_align_idempotence_constA) {
  SymGraph g;
  auto X = g.var();
  int A = 64;
  auto lhs = g.alignUp(g.alignUp(X, A), A);
  auto rhs = g.alignUp(X, A);
  EXPECT_EQ(NF(g, lhs), NF(g, rhs));
}

// AlignUp on a known multiple: alignUp(A*X, A) == A*X
TEST(symbolic, nonaffine_align_on_multiple_is_self) {
  SymGraph g;
  auto X = g.var();
  int A = 64;
  auto lhs = g.alignUp(g.mul(A, X), A);
  auto rhs = g.mul(A, X);
  EXPECT_EQ(NF(g, lhs), NF(g, rhs));
}

// Row-pitch equivalence under divisibility by constants:
// If A divisible by K=C*B, alignUp(W*K, A) == K * alignUp(W, A/K)
TEST(symbolic, nonaffine_row_pitch_align_equivalence_const_divisible) {
  SymGraph g;
  auto W = g.var();
  int C = 16, B = 1, A = 64, K = C * B; // A/K=4
  auto lhs = g.alignUp(g.mul(W, K), A);
  auto rhs = g.mul(g.alignUp(W, A / K), K);
  EXPECT_EQ(NF(g, lhs), NF(g, rhs));
}

// Bytes vs pixels align equivalence (constants):
// alignUp(W*K, A)/K == alignUp(W, A/K)
TEST(symbolic, nonaffine_align_equivalence_bytes_vs_pixels_const) {
  SymGraph g;
  auto W = g.var();
  int C = 16, B = 1, A = 64, K = C * B; // A/K=4
  auto lhs = g.div(g.alignUp(g.mul(W, K), A), K);
  auto rhs = g.alignUp(W, A / K);
  EXPECT_EQ(NF(g, lhs), NF(g, rhs));
}

// Mod composition with constants collapses to gcd:
// ((X % 12) % 8) == (X % 4)
TEST(symbolic, nonaffine_mod_constant_gcd_collapse) {
  SymGraph g;
  auto X = g.var();
  auto lhs = g.mod(g.mod(X, 12), 8);
  auto rhs = g.mod(X, 4);
  // They should both reduce to "X % 4" non-affine OR be syntactically equal;
  // Compare NF forms for robustness:
  EXPECT_EQ(NF(g, lhs), NF(g, rhs));
}

// Sum of multiples vanishes mod A (constant A), including a bare A term:
// (A*X + A*Y + A) % A == 0
TEST(symbolic, nonaffine_mod_sum_of_multiples_and_A_term_zero) {
  SymGraph g;
  auto X = g.var(), Y = g.var();
  int A = 32;
  auto e = g.mod(g.add(g.add(g.mul(A, X), g.mul(A, Y)), A), A);
  auto r = g.resolve(e);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(0, r.constant());
}

// Cancel then rescale by constant:
// (k*A*X) / A == k*X
TEST(symbolic, nonaffine_cancel_then_scale_by_const) {
  SymGraph g;
  auto A = g.var(), X = g.var();
  int k = 5;
  auto lhs = g.div(g.mul(g.mul(k, A), X), A);
  auto rhs = g.mul(k, X);
  EXPECT_EQ(NF(g, lhs), NF(g, rhs));
}
