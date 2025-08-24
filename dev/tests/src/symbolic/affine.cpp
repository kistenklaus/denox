#include "vkcnn/common/symbolic/SymGraph.hpp"
#include <gtest/gtest.h>

TEST(symbolic, constant_folding) {
  vkcnn::SymGraph graph;

  auto x = graph.resolve(graph.sub(5, 5));
  ASSERT_TRUE(!x.isSymbolic());
  EXPECT_EQ(x.value(), 0);

  auto y = graph.resolve(graph.add(7, 3));
  ASSERT_TRUE(!y.isSymbolic());
  EXPECT_EQ(y.value(), 10);

  auto z = graph.resolve(graph.mul(4, 3));
  ASSERT_TRUE(!z.isSymbolic());
  EXPECT_EQ(z.value(), 12);

  auto w = graph.resolve(graph.div(4, 3));
  ASSERT_TRUE(!w.isSymbolic());
  EXPECT_EQ(w.value(), 1);

  auto m = graph.resolve(graph.mod(12, 5));
  ASSERT_TRUE(!m.isSymbolic());
  EXPECT_EQ(m.value(), 2);
}

TEST(symbolic, affine_constant_folding) {
  vkcnn::SymGraph graph;
  auto A = graph.createParameter();
  auto x0 = graph.add(A, 10);
  auto x1 = graph.add(A, 2);
  auto x2 = graph.resolve(graph.sub(x0, x1));
  ASSERT_TRUE(!x2.isSymbolic());
  EXPECT_EQ(x2.value(), 8);
}

TEST(symbolic, affine_add_neutral) {
  vkcnn::SymGraph graph;
  auto A = graph.createParameter();
  auto x0 = graph.add(A, 0);
  EXPECT_EQ(A, x0);
}

TEST(symbolic, affine_sub_neutral) {
  vkcnn::SymGraph graph;
  auto A = graph.createParameter();
  auto x0 = graph.sub(A, 0);
  EXPECT_EQ(A, x0);
}

TEST(symbolic, affine_mul_neutral) {
  vkcnn::SymGraph graph;
  auto A = graph.createParameter();
  auto x0 = graph.mul(A, 1);
  EXPECT_EQ(A, x0);
}

TEST(symbolic, affine_div_neutral) {
  vkcnn::SymGraph graph;
  auto A = graph.createParameter();
  auto x0 = graph.div(A, 1);
  EXPECT_EQ(A, x0);
}

TEST(symbolic, do_not_optimize) {
  vkcnn::SymGraph graph;
  auto A = graph.createParameter();
  auto x0 = graph.add(A, 10);
  auto x1 = graph.add(A, 2);
  auto x2 = graph.sub(x0, x1);
  EXPECT_TRUE(x2.isSymbolic());
}

TEST(symbolic, affine_const_distributive_mul) {
  vkcnn::SymGraph graph;
  auto A = graph.createParameter();
  auto A2 = graph.mul(2, A);
  auto x = graph.sub(graph.mul(3, A), A2);
  EXPECT_EQ(x, A);
  auto x2 = graph.mul(A, 2);
  EXPECT_EQ(x2, A2);
}

TEST(symbolic, affine_div_elimination) {
  vkcnn::SymGraph graph;
  auto A = graph.createParameter();
  auto x0 = graph.resolve(graph.div(A, A));
  ASSERT_FALSE(x0.isSymbolic());
  EXPECT_EQ(x0.value(), 1);

  auto x2 = graph.resolve(graph.div(graph.mul(4, A), graph.add(A, A)));
  ASSERT_FALSE(x2.isSymbolic());
  EXPECT_EQ(x2.value(), 2);
}

TEST(symbolic, affine_trivial_sub_elimination) {

  vkcnn::SymGraph graph;
  auto A = graph.createParameter();

  auto x = graph.resolve(graph.sub(A, A));

  ASSERT_FALSE(x.isSymbolic());
  EXPECT_EQ(x.value(), 0);
}

TEST(symbolic, affine_sub_elimination) {

  vkcnn::SymGraph graph;
  auto A = graph.createParameter();
  auto B = graph.createParameter();
  auto C = graph.createParameter();

  auto A2 = graph.mul(2, A);
  auto B3 = graph.mul(3, B);
  auto C4 = graph.mul(4, C);

  auto AB = graph.add(A2, B3);
  auto ABC = graph.add(AB, C4);
  auto ABC5 = graph.add(ABC, 5);

  auto x = graph.resolve(graph.sub(ABC5, ABC5));

  ASSERT_FALSE(x.isSymbolic());
  EXPECT_EQ(x.value(), 0);
}

TEST(symbolic, affine_cross_term_elimination) {
  vkcnn::SymGraph graph;
  auto A = graph.createParameter();
  auto B = graph.createParameter();

  auto expr = graph.sub(graph.add(B, A), graph.add(A, B));
  auto x = graph.resolve(expr);

  ASSERT_FALSE(x.isSymbolic());
  EXPECT_EQ(x.value(), 0);
}

TEST(symbolic, affine_mixed_const_fold) {
  vkcnn::SymGraph graph;
  auto A = graph.createParameter();

  auto lhs = graph.add(graph.mul(2, A), 5);
  auto rhs = graph.add(graph.mul(2, A), 3);
  auto x = graph.resolve(graph.sub(lhs, rhs));

  ASSERT_FALSE(x.isSymbolic());
  EXPECT_EQ(x.value(), 2);
}

TEST(symbolic, affine_mul_zero) {
  vkcnn::SymGraph graph;
  auto A = graph.createParameter();

  auto x = graph.resolve(graph.mul(0, A));
  ASSERT_FALSE(x.isSymbolic());
  EXPECT_EQ(x.value(), 0);

  auto y = graph.resolve(graph.mul(A, 0));
  ASSERT_FALSE(y.isSymbolic());
  EXPECT_EQ(y.value(), 0);
}

TEST(symbolic, affine_div_constant_multiple) {
  vkcnn::SymGraph graph;
  auto A = graph.createParameter();
  auto A2 = graph.mul(A, 2);

  auto expr = graph.div(graph.mul(6, A), 3);
  auto x = graph.resolve(expr);

  EXPECT_TRUE(x.isSymbolic());
  EXPECT_EQ(A2, x);
}

TEST(symbolic, affine_mod_multiple) {
  vkcnn::SymGraph graph;
  auto A = graph.createParameter();

  auto expr = graph.mod(graph.mul(4, A), 2);
  auto x = graph.resolve(expr);

  ASSERT_FALSE(x.isSymbolic());
  EXPECT_EQ(x.value(), 0);
}

TEST(symbolic, affine_const_cancel) {
  vkcnn::SymGraph graph;
  auto A = graph.createParameter();

  auto lhs = graph.add(A, 5);
  auto rhs = graph.add(A, 5);
  auto x = graph.resolve(graph.sub(lhs, rhs));

  ASSERT_FALSE(x.isSymbolic());
  EXPECT_EQ(x.value(), 0);
}

TEST(symbolic, affine_nested_neutral) {
  vkcnn::SymGraph graph;
  auto A = graph.createParameter();

  auto expr = graph.div(graph.sub(graph.add(A, 0), 0), 1);
  EXPECT_EQ(expr, A);
}

TEST(symbolic, affine_div_fold_all_divisible) {
  vkcnn::SymGraph graph;
  auto A = graph.createParameter();

  auto expr = graph.div(graph.add(graph.mul(2, A), 4), 2);
  auto x = graph.resolve(expr);

  EXPECT_TRUE(x.isSymbolic()); // still affine
  // Should simplify to A + 2
  auto expected = graph.add(A, 2);
  EXPECT_EQ(x, expected);
}

TEST(symbolic, affine_mod_one) {
  vkcnn::SymGraph graph;
  auto A = graph.createParameter();

  auto expr = graph.mod(A, 1);
  auto x = graph.resolve(expr);

  ASSERT_FALSE(x.isSymbolic());
  EXPECT_EQ(x.value(), 0);
}

TEST(symbolic, affine_inner_const_fold) {
  vkcnn::SymGraph graph;
  auto A = graph.createParameter();

  auto inner = graph.sub(5, 3); // constant fold to 2
  auto expr = graph.add(A, inner);
  auto x = graph.resolve(expr);

  EXPECT_TRUE(x.isSymbolic());
  auto expected = graph.add(A, 2);
  EXPECT_EQ(x, expected);
}

TEST(symbolic, affine_mod_scaled_same_base) {
  vkcnn::SymGraph g;
  auto A = g.createParameter();
  auto E = g.add(A, 1); // E = A+1

  auto x1 = g.resolve(g.mod(g.mul(6, E), g.mul(4, E))); // (6 % 4) * E = 2E
  auto exp1 = g.mul(2, E);
  EXPECT_EQ(x1, exp1);

  auto x2 = g.resolve(g.mod(g.mul(3, E), g.mul(2, E))); // (3 % 2) * E = 1E
  EXPECT_EQ(x2, E);
}

TEST(symbolic, affine_div_all_terms_divisible) {
  vkcnn::SymGraph g;
  auto A = g.createParameter();
  auto E = g.add(g.mul(4, A), 8);  // 4A + 8
  auto q = g.resolve(g.div(E, 4)); // -> A + 2
  auto expected = g.add(A, 2);
  EXPECT_EQ(q, expected);
}

TEST(symbolic, affine_merge_same_symbol) {
  vkcnn::SymGraph g;
  auto A = g.createParameter();
  auto lhs = g.add(g.add(A, A), A);
  auto rhs = g.mul(3, A);
  auto d = g.resolve(g.sub(lhs, rhs));
  ASSERT_FALSE(d.isSymbolic());
  EXPECT_EQ(d.value(), 0);
}

TEST(symbolic, affine_add_assoc_comm) {
  vkcnn::SymGraph g;
  auto A = g.createParameter(), B = g.createParameter(),
       C = g.createParameter();
  auto e1 = g.add(A, g.add(B, C));
  auto e2 = g.add(g.add(C, A), B);
  auto d = g.resolve(g.sub(e1, e2));
  ASSERT_FALSE(d.isSymbolic());
  EXPECT_EQ(d.value(), 0);
}

TEST(symbolic, affine_mod_same_base_scaled) {
  vkcnn::SymGraph g;
  auto A = g.createParameter();
  auto E = g.add(A, 1); // E = A+1

  auto r1 = g.resolve(g.mod(g.mul(6, E), g.mul(4, E))); // (6%4)=2 → 2E
  EXPECT_EQ(r1, g.mul(2, E));

  auto r2 = g.resolve(g.mod(g.mul(6, E), g.mul(3, E))); // (6%3)=0 → 0
  ASSERT_FALSE(r2.isSymbolic());
  EXPECT_EQ(r2.value(), 0);
}

TEST(symbolic, affine_cancel_inside) {
  vkcnn::SymGraph g;
  auto A = g.createParameter();
  auto e = g.sub(g.add(A, 5), g.add(A, 5));
  auto r = g.resolve(e);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(r.value(), 0);
}

TEST(symbolic, affine_mod_same_base_zero_remainder) {
  vkcnn::SymGraph g;
  auto A = g.createParameter();
  auto E = g.add(A, 5);
  auto r = g.resolve(g.mod(g.mul(8, E), g.mul(4, E))); // 8%4=0
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(r.value(), 0);
}

TEST(symbolic, affine_mod_same_base_unit_remainder) {
  vkcnn::SymGraph g;
  auto A = g.createParameter();
  auto E = g.add(A, 2);
  auto r = g.resolve(g.mod(g.mul(5, E), g.mul(4, E))); // 5%4=1 → E
  EXPECT_EQ(r, E);
}

TEST(symbolic, affine_mod_constant_only_constant_residue) {
  vkcnn::SymGraph g;
  auto A = g.createParameter();
  auto e = g.add(g.mul(6, A), 5);  // 6A + 5
  auto r = g.resolve(g.mod(e, 3)); // 6A%3=0, 5%3=2
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(r.value(), 2);
}

TEST(symbolic, affine_div_all_terms_divisible_with_zero_const) {
  vkcnn::SymGraph g;
  auto A = g.createParameter();
  auto q = g.resolve(g.div(g.mul(8, A), 4)); // -> 2A
  EXPECT_EQ(q, g.mul(2, A));
}

TEST(symbolic, affine_add_large_permutation_equal) {
  vkcnn::SymGraph g;
  auto A = g.createParameter(), B = g.createParameter(),
       C = g.createParameter(), D = g.createParameter();
  auto e1 = g.add(g.add(A, g.add(C, B)), D);
  auto e2 = g.add(g.add(D, A), g.add(B, C));
  auto d = g.resolve(g.sub(e1, e2));
  ASSERT_FALSE(d.isSymbolic());
  EXPECT_EQ(d.value(), 0);
}

TEST(symbolic, affine_mod_same_base_zero_scale_canonical) {
  vkcnn::SymGraph g;
  auto A = g.createParameter();
  auto E = g.add(A, 3);
  auto r = g.resolve(g.mod(g.mul(6, E), g.mul(3, E))); // 6%3=0
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(r.value(), 0);
}

TEST(symbolic, affine_linearity_holds) {
  vkcnn::SymGraph g;
  auto A = g.createParameter(), B = g.createParameter();
  auto left = g.mul(5, g.add(g.add(A, 2), B));             // 5*(A+2+B)
  auto right = g.add(g.add(g.mul(5, A), 10), g.mul(5, B)); // 5A+10+5B
  auto d = g.resolve(g.sub(left, right));
  ASSERT_FALSE(d.isSymbolic());
  EXPECT_EQ(d.value(), 0);
}

TEST(symbolic, affine_ceildiv) {
  vkcnn::SymGraph g;
  auto A = g.createParameter();

  auto A2 = g.mul(A, 2);
  auto rhs = A;
  auto lhs = g.div(g.add(A2, 2 - 1), 2);
  EXPECT_EQ(rhs, lhs);
}

TEST(symbolic, affine_div_with_common_factor) {
  vkcnn::SymGraph g;
  auto A = g.createParameter();
  auto B = g.createParameter();

  auto lhs = g.div(g.add(g.add(g.mul(4, A), g.mul(2, B)), 3), 2);
  auto rhs = g.add(g.add(g.mul(2, A), g.mul(1, B)), 1);
  EXPECT_EQ(rhs, lhs);
}

TEST(symbolic, affine_div_with_neg_constant) {
  vkcnn::SymGraph g;
  auto A = g.createParameter();

  auto lhs = g.div(g.sub(g.mul(2, A), 1), 2);
  auto rhs = g.sub(A, 1);
  EXPECT_EQ(rhs, lhs);
}

TEST(symbolic, affine_div_with_pos_constant) {
  vkcnn::SymGraph g;
  auto A = g.createParameter();

  auto lhs = g.div(g.add(g.mul(2, A), 1), 2);
  auto rhs = A;
  EXPECT_EQ(rhs, lhs);
}

TEST(symbolic, affine_floordiv_coefficients) {
  vkcnn::SymGraph g;
  auto A = g.createParameter();
  auto B = g.createParameter();
  auto lhs = g.div(g.add(g.add(g.mul(6, A), g.mul(10, B)), 7), 2);
  auto rhs = g.add(g.add(g.mul(3, A), g.mul(5, B)), 3);
  EXPECT_EQ(rhs, lhs);
}

TEST(symbolic, affine_div_exact_no_residual) {
  vkcnn::SymGraph g;
  auto A = g.createParameter(); auto B = g.createParameter();
  // (6A + 10B + 8)/2 -> 3A + 5B + 4
  auto lhs = g.div(g.add(g.add(g.mul(6, A), g.mul(10, B)), 8), 2);
  auto rhs = g.add(g.add(g.mul(3, A), g.mul(5, B)), 4);
  EXPECT_EQ(rhs, lhs);
}

TEST(symbolic, affine_div_constant_remainder_only) {
  vkcnn::SymGraph g;
  auto A = g.createParameter(); auto B = g.createParameter();
  // (6A + 10B + 7)/2 -> 3A + 5B + 3   (Euclidean floor on constant)
  auto lhs = g.div(g.add(g.add(g.mul(6, A), g.mul(10, B)), 7), 2);
  auto rhs = g.add(g.add(g.mul(3, A), g.mul(5, B)), 3);
  EXPECT_EQ(rhs, lhs);
}

TEST(symbolic, affine_div_negative_constant_euclidean) {
  vkcnn::SymGraph g;
  auto A = g.createParameter();
  // (2A - 1)/2 -> A - 1
  auto lhs = g.div(g.sub(g.mul(2, A), 1), 2);
  auto rhs = g.sub(A, 1);
  EXPECT_EQ(rhs, lhs);
}

TEST(symbolic, affine_div_k_times_w_plus_k_minus_one_over_k) {
  vkcnn::SymGraph g;
  auto W = g.createParameter();
  std::size_t k = 5;
  // (k*W + (k-1))/k -> W
  auto lhs = g.div(g.add(g.mul(k, W), k - 1), k);
  EXPECT_EQ(W, lhs);
}

