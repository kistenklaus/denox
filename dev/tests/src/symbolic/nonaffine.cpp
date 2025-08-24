#include "vkcnn/common/symbolic/SymGraph.hpp"
#include <gtest/gtest.h>

#define EXPECT_SEMANTIC_EQ(G, LHS, RHS)                                        \
  do {                                                                         \
    auto __d = (G).resolve((G).sub((LHS), (RHS)));                             \
    ASSERT_FALSE(__d.isSymbolic());                                            \
    EXPECT_EQ(__d.value(), 0);                                                 \
  } while (0)

// Commutativity: A*B == B*A  (should already pass)
TEST(symbolic, nonaffine_mul_commutative_semantic) {
  vkcnn::SymGraph g;
  auto A = g.createParameter(), B = g.createParameter();
  EXPECT_SEMANTIC_EQ(g, g.mul(A, B), g.mul(B, A));
}

// Associativity: (A*B)*C == A*(B*C)  (semantic truth; may fail until you
// canonicalize products)
TEST(symbolic, nonaffine_mul_associative_semantic) {
  vkcnn::SymGraph g;
  auto A = g.createParameter(), B = g.createParameter(),
       C = g.createParameter();
  EXPECT_SEMANTIC_EQ(g, g.mul(g.mul(A, B), C), g.mul(A, g.mul(B, C)));
}

// Left distributivity: A*(B+C) == A*B + A*C
TEST(symbolic, nonaffine_left_distributive_semantic) {
  vkcnn::SymGraph g;
  auto A = g.createParameter(), B = g.createParameter(),
       C = g.createParameter();
  EXPECT_SEMANTIC_EQ(g, g.mul(A, g.add(B, C)), g.add(g.mul(A, B), g.mul(A, C)));
}

// Right distributivity: (A+B)*C == A*C + B*C
TEST(symbolic, nonaffine_right_distributive_semantic) {
  vkcnn::SymGraph g;
  auto A = g.createParameter(), B = g.createParameter(),
       C = g.createParameter();
  EXPECT_SEMANTIC_EQ(g, g.mul(g.add(A, B), C), g.add(g.mul(A, C), g.mul(B, C)));
}

// Scalar commutation & associativity with product: k*(A*B) == (k*A)*B ==
// A*(k*B)
TEST(symbolic, nonaffine_scalar_product_semantic) {
  vkcnn::SymGraph g;
  auto A = g.createParameter(), B = g.createParameter();
  EXPECT_SEMANTIC_EQ(g, g.mul(3, g.mul(A, B)), g.mul(g.mul(3, A), B));
  EXPECT_SEMANTIC_EQ(g, g.mul(3, g.mul(A, B)), g.mul(A, g.mul(3, B)));
}

// Neutral/absorbing elements: 1*E == E, E*1 == E, 0*E == 0, E*0 == 0
TEST(symbolic, nonaffine_neutral_absorbing_semantic) {
  vkcnn::SymGraph g;
  auto A = g.createParameter(), B = g.createParameter();
  auto E = g.add(A, B); // any positive expression
  EXPECT_SEMANTIC_EQ(g, g.mul(1, E), E);
  EXPECT_SEMANTIC_EQ(g, g.mul(E, 1), E);
  auto z1 = g.resolve(g.mul(0, E));
  ASSERT_FALSE(z1.isSymbolic());
  EXPECT_EQ(z1.value(), 0);
  auto z2 = g.resolve(g.mul(E, 0));
  ASSERT_FALSE(z2.isSymbolic());
  EXPECT_EQ(z2.value(), 0);
}

// Same-base scaled mod: (k·E) % (m·E) == (k%m)·E
TEST(symbolic, nonaffine_mod_same_base_scaled_semantic) {
  vkcnn::SymGraph g;
  auto A = g.createParameter(), B = g.createParameter();
  auto E = g.add(g.mul(A, B), 1); // ensure E > 0
  EXPECT_SEMANTIC_EQ(g, g.mod(g.mul(5, E), g.mul(3, E)), g.mul(5 % 3, E)); // 2E
  auto r0 = g.resolve(g.mod(g.mul(6, E), g.mul(3, E)));
  ASSERT_FALSE(r0.isSymbolic());
  EXPECT_EQ(r0.value(), 0);
}

// Mod by self: E % E == 0
TEST(symbolic, nonaffine_mod_self_zero_semantic) {
  vkcnn::SymGraph g;
  auto A = g.createParameter(), B = g.createParameter();
  auto E = g.add(g.mul(A, B), g.add(A, 2));
  auto r = g.resolve(g.mod(E, E));
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(r.value(), 0);
}

// Mod by a factor: (A*B) % A == 0
TEST(symbolic, nonaffine_mod_by_factor_zero_semantic) {
  vkcnn::SymGraph g;
  auto A = g.createParameter(), B = g.createParameter();
  auto r = g.resolve(g.mod(g.mul(A, B), A));
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(r.value(), 0);
}

// Exact cancel in division (requires positivity; semantic truth)
// May fail until you implement factor-cancel in div.
TEST(symbolic, nonaffine_exact_div_cancel_semantic) {
  vkcnn::SymGraph g;
  auto A = g.createParameter(), B = g.createParameter();
  EXPECT_SEMANTIC_EQ(g, g.div(g.mul(A, B), A), B);
}

// Scaled exact cancel: (k·E)/k == E (k>0)
TEST(symbolic, nonaffine_scaled_exact_div_cancel_semantic) {
  vkcnn::SymGraph g;
  auto A = g.createParameter();
  auto E = g.add(g.mul(4, A), 6);
  EXPECT_SEMANTIC_EQ(g, g.div(g.mul(5, E), 5), E);
}

// GCD-scaled div same base: (k·E)/(m·E) == floor(k/m)
TEST(symbolic, nonaffine_gcd_scaled_div_same_base_semantic) {
  vkcnn::SymGraph g;
  auto A = g.createParameter(), B = g.createParameter();
  auto E = g.add(g.mul(A, B), 1);
  auto q = g.resolve(g.div(g.mul(7, E), g.mul(3, E))); // floor(7/3)=2
  ASSERT_FALSE(q.isSymbolic());
  EXPECT_EQ(q.value(), 2);
}


// A*(B + C) == A*B + A*C
TEST(symbolic, nonaffine_left_distributivity) {
  vkcnn::SymGraph g;
  auto A = g.createParameter();
  auto B = g.createParameter();
  auto C = g.createParameter();

  auto lhs = g.mul(A, g.add(B, C));
  auto rhs = g.add(g.mul(A, B), g.mul(A, C));
  EXPECT_EQ(lhs, rhs);
}

// Same as above but with (C + B) to catch any order-sensitivity bugs
TEST(symbolic, nonaffine_left_distributivity_comm_order) {
  vkcnn::SymGraph g;
  auto A = g.createParameter();
  auto B = g.createParameter();
  auto C = g.createParameter();

  auto lhs = g.mul(A, g.add(C, B));  // flipped order inside sum
  auto rhs = g.add(g.mul(A, B), g.mul(A, C));
  EXPECT_EQ(lhs, rhs);
}

// (A + B)*C == A*C + B*C  (right-distributivity)
TEST(symbolic, nonaffine_right_distributivity) {
  vkcnn::SymGraph g;
  auto A = g.createParameter();
  auto B = g.createParameter();
  auto C = g.createParameter();

  auto lhs = g.mul(g.add(A, B), C);
  auto rhs = g.add(g.mul(A, C), g.mul(B, C));
  EXPECT_EQ(lhs, rhs);
}

// A*(B + C) == A*B + A*C  (left distributivity – variant 2)
TEST(symbolic, nonaffine_left_distributivity_nested) {
  vkcnn::SymGraph g;
  auto A=g.createParameter(), B=g.createParameter(), C=g.createParameter(), D=g.createParameter();
  auto lhs = g.mul(A, g.add(B, g.add(C, D)));
  auto rhs = g.add(g.add(g.mul(A,B), g.mul(A,C)), g.mul(A,D));
  auto d = g.resolve(g.sub(lhs, rhs));
  ASSERT_FALSE(d.isSymbolic());
  EXPECT_EQ(d.value(), 0);
}

// (A + B)*C == A*C + B*C  (right distributivity – variant 2)
TEST(symbolic, nonaffine_right_distributivity_nested) {
  vkcnn::SymGraph g;
  auto A=g.createParameter(), B=g.createParameter(), C=g.createParameter(), D=g.createParameter();
  auto lhs = g.mul(g.add(A, g.add(B, C)), D);
  auto rhs = g.add(g.add(g.mul(A,D), g.mul(B,D)), g.mul(C,D));
  EXPECT_EQ(lhs, rhs);
}

// Full bilinear expansion: (A + B) * (C + D) == AC + AD + BC + BD
TEST(symbolic, nonaffine_bilinear_expansion) {
  vkcnn::SymGraph g;
  auto A=g.createParameter(), B=g.createParameter(), C=g.createParameter(), D=g.createParameter();
  auto lhs = g.mul(g.add(A,B), g.add(C,D));
  auto rhs = g.add(g.add(g.mul(A,C), g.mul(A,D)), g.add(g.mul(B,C), g.mul(B,D)));
  EXPECT_EQ(lhs, rhs);
}

// Distribute with constants inside sum: A*(B + 2) == A*B + 2*A
TEST(symbolic, nonaffine_left_distributivity_with_const) {
  vkcnn::SymGraph g;
  auto A=g.createParameter(), B=g.createParameter();
  auto lhs = g.mul(A, g.add(B, 2));
  auto rhs = g.add(g.mul(A,B), g.mul(2, A));
  EXPECT_EQ(lhs, rhs);
}

// Distribute with constants (right): (A + 2)*B == A*B + 2*B
TEST(symbolic, nonaffine_right_distributivity_with_const) {
  vkcnn::SymGraph g;
  auto A=g.createParameter(), B=g.createParameter();
  auto lhs = g.mul(g.add(A, 2), B);
  auto rhs = g.add(g.mul(A,B), g.mul(2, B));
  EXPECT_EQ(lhs, rhs);
}

// Factor common product: A*B + A*C == A*(B + C)
TEST(symbolic, nonaffine_factor_common_left) {
  vkcnn::SymGraph g;
  auto A=g.createParameter(), B=g.createParameter(), C=g.createParameter();
  auto lhs = g.add(g.mul(A,B), g.mul(A,C));
  auto rhs = g.mul(A, g.add(B,C));
  EXPECT_EQ(lhs, rhs);
}

// Factor common product with scalar: 2*A*B + 2*A*C == 2*A*(B + C)
TEST(symbolic, nonaffine_factor_common_with_scalar) {
  vkcnn::SymGraph g;
  auto A=g.createParameter(), B=g.createParameter(), C=g.createParameter();
  auto lhs = g.add(g.mul(g.mul(2,A), B), g.mul(g.mul(2,A), C));
  auto rhs = g.mul(g.mul(2,A), g.add(B,C));
  EXPECT_EQ(lhs, rhs);
}

// Combine distributivity across three terms: A*(B + C) + A*D == A*(B + C + D)
TEST(symbolic, nonaffine_distribute_three_terms) {
  vkcnn::SymGraph g;
  auto A=g.createParameter(), B=g.createParameter(), C=g.createParameter(), D=g.createParameter();
  auto lhs = g.add(g.mul(A, g.add(B,C)), g.mul(A, D));
  auto rhs = g.mul(A, g.add(g.add(B,C), D));
  EXPECT_EQ(lhs, rhs);
}

// Pull scalar through sum of products: k*(A*B + A*C) == A*(k*B + k*C)
TEST(symbolic, nonaffine_scalar_through_sum_of_products) {
  vkcnn::SymGraph g;
  auto A=g.createParameter(), B=g.createParameter(), C=g.createParameter();
  auto k = 3;
  auto lhs = g.mul(k, g.add(g.mul(A,B), g.mul(A,C)));
  auto rhs = g.mul(A, g.add(g.mul(k,B), g.mul(k,C)));
  EXPECT_EQ(lhs, rhs);
}

// Mod with same base wrapped in sums/products: (k·E)%(m·E) = (k%m)·E, with E nontrivial
TEST(symbolic, nonaffine_mod_same_base_scaled_complex_E) {
  vkcnn::SymGraph g;
  auto A=g.createParameter(), B=g.createParameter(), C=g.createParameter();
  auto E = g.mul(g.add(A,1), g.add(B,C));   // E = (A+1)*(B+C)
  auto lhs = g.mod(g.mul(7, E), g.mul(3, E));
  auto rhs = g.mul(7 % 3, E);               // 2*E
  EXPECT_EQ(lhs, rhs);
}

// A*(B + C) == A*B + A*C     (left distributivity)  — likely FAIL until you distribute into sums
TEST(symbolic, nonaffine_left_distributivity_structural) {
  vkcnn::SymGraph g;
  auto A=g.createParameter(), B=g.createParameter(), C=g.createParameter();

  auto lhs = g.mul(A, g.add(B, C));
  auto rhs = g.add(g.mul(A, B), g.mul(A, C));
  EXPECT_EQ(lhs, rhs);
}

// (A + B)*C == A*C + B*C      (right distributivity) — you said this currently FAILs
TEST(symbolic, nonaffine_right_distributivity_structural) {
  vkcnn::SymGraph g;
  auto A=g.createParameter(), B=g.createParameter(), C=g.createParameter();

  auto lhs = g.mul(g.add(A, B), C);
  auto rhs = g.add(g.mul(A, C), g.mul(B, C));
  EXPECT_EQ(lhs, rhs);
}

// (A + B)*(C + D) == AC + AD + BC + BD  — full bilinear expansion
TEST(symbolic, nonaffine_bilinear_expansion_structural) {
  vkcnn::SymGraph g;
  auto A=g.createParameter(), B=g.createParameter(), C=g.createParameter(), D=g.createParameter();

  auto lhs = g.mul(g.add(A,B), g.add(C,D));
  auto rhs = g.add(g.add(g.mul(A,C), g.mul(A,D)),
                   g.add(g.mul(B,C), g.mul(B,D)));
  EXPECT_EQ(lhs, rhs);
}

// Factor out common left factor: A*B + A*C == A*(B + C)
TEST(symbolic, nonaffine_factor_common_left_structural) {
  vkcnn::SymGraph g;
  auto A=g.createParameter(), B=g.createParameter(), C=g.createParameter();

  auto lhs = g.add(g.mul(A,B), g.mul(A,C));
  auto rhs = g.mul(A, g.add(B,C));
  EXPECT_EQ(lhs, rhs);
}

// Factor out common right factor: B*A + C*A == (B + C)*A
TEST(symbolic, nonaffine_factor_common_right_structural) {
  vkcnn::SymGraph g;
  auto A=g.createParameter(), B=g.createParameter(), C=g.createParameter();

  auto lhs = g.add(g.mul(B,A), g.mul(C,A));
  auto rhs = g.mul(g.add(B,C), A);
  EXPECT_EQ(lhs, rhs);
}

// Merge identical product terms: (AB + BA) == 2*(AB)
TEST(symbolic, nonaffine_merge_identical_products_structural) {
  vkcnn::SymGraph g;
  auto A=g.createParameter(), B=g.createParameter();

  auto AB = g.mul(A,B);
  auto BA = g.mul(B,A);
  auto lhs = g.add(AB, BA);
  auto rhs = g.mul(2, AB);
  EXPECT_EQ(lhs, rhs);
}

// Scalar movement through product: 3*(A*B) == (3*A)*B == A*(3*B)  — should PASS if you commute scalars
TEST(symbolic, nonaffine_scalar_move_through_product_structural) {
  vkcnn::SymGraph g;
  auto A=g.createParameter(), B=g.createParameter();

  auto left  = g.mul(3, g.mul(A,B));
  auto mid   = g.mul(g.mul(3,A), B);
  auto right = g.mul(A, g.mul(3,B));
  EXPECT_EQ(left, mid);
  EXPECT_EQ(left, right);
}

// Exact cancel in division: (A*(B+C))/A == (B+C)  — likely FAIL until you implement factor-cancel
TEST(symbolic, nonaffine_div_cancel_common_factor_sum_structural) {
  vkcnn::SymGraph g;
  auto A=g.createParameter(), B=g.createParameter(), C=g.createParameter();

  auto lhs = g.div(g.mul(A, g.add(B,C)), A);
  auto rhs = g.add(B,C);
  EXPECT_EQ(lhs, rhs);
}

// Scaled exact cancel: (6*(B+C))/6 == B+C
TEST(symbolic, nonaffine_div_cancel_scalar_sum_structural) {
  vkcnn::SymGraph g;
  auto B=g.createParameter(), C=g.createParameter();

  auto lhs = g.div(g.mul(6, g.add(B,C)), 6);
  auto rhs = g.add(B,C);
  EXPECT_EQ(lhs, rhs);
}

// GCD-scaled div with same base: (k·E)/(m·E) == floor(k/m)
// Compare against an explicit constant: build 2 as div(4,2)
TEST(symbolic, nonaffine_div_same_base_scale_structural) {
  vkcnn::SymGraph g;
  auto A=g.createParameter(), B=g.createParameter();
  auto E = g.add(g.mul(A,B), 1);  // E > 0

  auto lhs = g.div(g.mul(7, E), g.mul(3, E));   // expect 2
  auto rhs = g.div(4, 2);                       // constant-2 node
  EXPECT_EQ(lhs, rhs);
}

// Mod same base scaled: (k·E) % (m·E) == (k%m)·E
TEST(symbolic, nonaffine_mod_same_base_scaled_structural) {
  vkcnn::SymGraph g;
  auto A=g.createParameter(), B=g.createParameter();
  auto E = g.add(g.mul(A,B), 1);

  auto lhs = g.mod(g.mul(5, E), g.mul(3, E)); // expect 2*E
  auto rhs = g.mul(2, E);
  EXPECT_EQ(lhs, rhs);

  auto lhs0 = g.mod(g.mul(6, E), g.mul(3, E)); // expect 0
  auto rhs0 = g.mul(0, E);                     // canonical zero node
  EXPECT_EQ(lhs0, rhs0);
}

// Mod by factor: (A*B) % A == 0  — you observed FAIL now; keep as target
TEST(symbolic, nonaffine_mod_by_factor_zero_structural) {
  vkcnn::SymGraph g;
  auto A=g.createParameter(), B=g.createParameter();
  auto lhs = g.mod(g.mul(A,B), A);
  auto rhs = g.mul(0, A);                      // zero node
  EXPECT_EQ(lhs, rhs);
}

// Reconstruction identity: X == m*div(X,m) + (X % m) — structural goal; may FAIL until you canonicalize
TEST(symbolic, nonaffine_div_mod_reconstruction_structural) {
  vkcnn::SymGraph g;
  auto A=g.createParameter(), B=g.createParameter();
  auto X = g.mul(g.add(A,1), g.add(B,2));
  const unsigned m = 5;

  auto lhs = X;
  auto rhs = g.add(g.mul(m, g.div(X, m)), g.mod(X, m));
  EXPECT_EQ(lhs, rhs);
}


// Ceil(k*a, a) == k
TEST(symbolic, nonaffine_ceildiv_exact_multiple) {
  vkcnn::SymGraph g;
  auto a=g.createParameter(), k=g.createParameter();
  auto ceildiv = g.div(g.add(g.mul(k, a), g.sub(a, 1)), a);
  EXPECT_EQ(ceildiv, k);
}

// Ceil(a*b + (a-1), a) == b + 1
TEST(symbolic, nonaffine_ceildiv_one_more_block) {
  vkcnn::SymGraph g;
  auto a=g.createParameter(), b=g.createParameter();
  auto ceildiv = g.div(g.add(g.mul(a, b), g.sub(a, 1)), a);
  auto expect  = g.add(b, 1);
  EXPECT_EQ(ceildiv, expect);
}

// Ceil((a*(b+c)), a) == (b+c)
TEST(symbolic, nonaffine_ceildiv_cancel_factor_sum) {
  vkcnn::SymGraph g;
  auto a=g.createParameter(), b=g.createParameter(), c=g.createParameter();
  auto ceildiv = g.div(g.add(g.mul(a, g.add(b,c)), g.sub(a,1)), a);
  auto expect  = g.add(b, c); // since a*(b+c) already multiple of a
  EXPECT_EQ(ceildiv, expect);
}
