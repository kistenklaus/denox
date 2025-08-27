#include "vkcnn/common/symbolic/SymGraph.hpp"
#include <gtest/gtest.h>

TEST(symbolic, unsolvable_div_not_divisible_bails) {
  vkcnn::SymGraph g;
  auto A = g.var();
  auto e = g.add(g.mul(6, A), 8);  // 6A+8
  auto q = g.resolve(g.div(e, 4)); // 6A/4 not integer → symbolic
  EXPECT_TRUE(q.isSymbolic());
}

TEST(symbolic, unsolvable_div_by_var_nontrivial) {
  vkcnn::SymGraph g;
  auto A = g.var();
  // (2A + 1) / A  -> not a constant (your A/A special-case doesn't apply)
  auto q = g.resolve(g.div(g.add(g.mul(2, A), 1), A));
  EXPECT_TRUE(q.isSymbolic());
}

TEST(symbolic, unsolvable_div_const_not_all_divisible) {
  vkcnn::SymGraph g;
  auto A = g.var();
  // (A + 1) / 2  -> some terms not divisible by 2 => bail
  auto q = g.resolve(g.div(g.add(A, 1), 2));
  EXPECT_TRUE(q.isSymbolic());
}

TEST(symbolic, unsolvable_mod_const_not_all_divisible) {
  vkcnn::SymGraph g;
  auto A = g.var();
  // (A + 1) % 2  -> coef 1 not divisible by 2 => bail
  auto r = g.resolve(g.mod(g.add(A, 1), 2));
  EXPECT_TRUE(r.isSymbolic());
}

TEST(symbolic, unsolvable_const_mod_affine_bails) {
  vkcnn::SymGraph g;
  auto A = g.var();
  // 7 % (A + 2)  -> rhs non-constant => bail in Z-affine
  auto r = g.resolve(g.mod(7, g.add(A, 2)));
  EXPECT_TRUE(r.isSymbolic());
}

TEST(symbolic, unsolvable_affine_mod_affine_different_base_bails) {
  vkcnn::SymGraph g;
  auto A = g.var();
  // (A + 1) % (A + 3)  -> bases differ, gcd-rule doesn't apply => bail
  auto r = g.resolve(g.mod(g.add(A, 1), g.add(A, 3)));
  EXPECT_TRUE(r.isSymbolic());
}

TEST(symbolic, unsolvable_div_const_not_divisible_some_terms) {
  vkcnn::SymGraph g;
  auto A = g.var();
  // (4A + 2) / 3  -> coefficients not all divisible by 3 => bail
  auto q = g.resolve(g.div(g.add(g.mul(4, A), 2), 3));
  EXPECT_TRUE(q.isSymbolic());
}

TEST(symbolic, unsolvable_compound_bail_div_then_mod) {
  vkcnn::SymGraph g;
  auto A = g.var();
  // (A + 2)/2  already bails -> symbol; then %2 should also bail in Z-affine
  auto r = g.resolve(g.mod(g.div(g.add(A, 2), 2), 2));
  EXPECT_TRUE(r.isSymbolic());
}

TEST(symbolic, unsolvable_mod_same_base_nonzero_remainder_is_affine_not_const) {
  vkcnn::SymGraph g;
  auto A = g.var();
  auto E = g.add(A, 1); // affine base
  // (3E) % (2E) = (3%2) * E = E  -> symbolic (non-constant) but simplified
  auto r = g.resolve(g.mod(g.mul(3, E), g.mul(2, E)));
  EXPECT_EQ(r, E);
  EXPECT_TRUE(r.isSymbolic());
}

TEST(symbolic, unsolvable_parity) {
  vkcnn::SymGraph g;
  auto A = g.var();
  auto r = g.resolve(g.mod(g.add(A, 1), 2)); // depends on A parity
  EXPECT_TRUE(r.isSymbolic());
}

TEST(symbolic, unsolvable_no_cancel_when_not_multiple) {
  vkcnn::SymGraph g; auto A=g.var(), B=g.var();
  auto lhs = g.div(g.add(B, g.sub(A,1)), A);            // (B + A - 1)/A
  EXPECT_NE(B, lhs);                                     // not equal to B
}

TEST(symbolic, unsolvable_no_factorization_AB_plus_A_over_AC) {
  vkcnn::SymGraph g; auto A=g.var(), B=g.var(), C=g.var();
  auto lhs = g.div(g.add(g.mul(A,B), A), g.mul(A,C));    // (AB + A)/(AC)
  auto wrong = g.div(g.add(1, B), C);                    // (1+B)/C  (needs factoring A)
  EXPECT_NE(wrong, lhs);                                 // we don't factor
}

TEST(symbolic, unsolvable_not_zero_A_over_AB) {
  vkcnn::SymGraph g; auto A=g.var(), B=g.var();
  auto expr = g.div(A, g.mul(A,B));                      // A/(A*B) == 1/B, not 0
  EXPECT_NE(g.mul(0, A), expr);
  EXPECT_EQ(g.div(1, B), expr);
}

TEST(symbolic, unsolvable_not_B_AB_over_AC) {
  vkcnn::SymGraph g; auto A=g.var(), B=g.var(), C=g.var();
  auto lhs = g.div(g.mul(A,B), g.mul(A,C));              // AB/AC
  EXPECT_NE(B, lhs);                                     // only B/C is correct
}

TEST(symbolic, unsolvable_div_const_over_symbol_not_zero) {
  vkcnn::SymGraph g; auto X=g.var();
  auto expr = g.div(1, X);
  EXPECT_NE(g.mul(0, X), expr);                          // 1/X is not 0 in general
}

TEST(symbolic, unsolvable_no_discharge_wrong_residual) {
  vkcnn::SymGraph g; auto A=g.var(), B=g.var(), C=g.var();
  auto lhs = g.div(g.sub(C,1), g.mul(A,B));              // (C - 1)/(AB)  (residual is C-1, not AB-1)
  EXPECT_NE(g.mul(0, A), lhs);                           // must not drop to 0
}

TEST(symbolic, unsolvable_disjoint_no_cancel) {
  vkcnn::SymGraph g; auto A=g.var(), B=g.var(), D=g.var();
  auto lhs = g.div(g.mul(A,B), D);                       // (A*B)/D, no common factor
  auto wrong = g.div(B, D);                              // pretending A cancels
  EXPECT_NE(wrong, lhs);
}

TEST(symbolic, unsolvable_nested_div_not_flattened_to_A_over_B) {
  vkcnn::SymGraph g; auto A=g.var(), B=g.var();
  auto lhs = g.div(g.div(A, B), B);                      // (A/B)/B
  EXPECT_NE(g.div(A, B), lhs);                           // must be A/(B*B)
  EXPECT_EQ(g.div(A, g.mul(B,B)), lhs);
}

TEST(symbolic, unsolvable_prod_gcd_no_underflow_on_multiplicity) {
  vkcnn::SymGraph g; auto A=g.var(), C=g.var();
  auto lhs = g.div(g.mul(A, C), g.mul(C, C));            // (A*C)/(C*C) == A/C
  EXPECT_EQ(g.div(A, C), lhs);
  EXPECT_NE(A, lhs);                                     // not A (since C remains)
}

TEST(symbolic, unsolvable_modd_depends_on_symbol) {
  vkcnn::SymGraph g; auto A=g.var(), B=g.var();
  // (4A + 3B + 1) % 2  depends on B (3%2=1) -> cannot fold to affine
  auto e = g.add(g.add(g.mul(4, A), g.mul(3, B)), 1);
  auto m = g.mod(e, 2);
  // Expect: still symbolic (i.e., affine_mod returned nullopt and you created a Mod node)
  auto r = g.resolve(m);
  EXPECT_TRUE(r.isSymbolic()); // or however you check non-constant here
}

TEST(symbolic, unsolvable_mod_affine_sum_not_factored) {
  vkcnn::SymGraph g; auto A=g.var(), B=g.var();
  auto expr = g.mod(g.add(g.mul(A,B), A), A);        // (A*B + A) % A == 0 mathematically,
                                                     // but we are not factoring sums here.
  auto zero = g.mul(0, A);
  EXPECT_NE(zero, expr);
}

TEST(symbolic, unsolvable_mod_disjoint_products) {
  vkcnn::SymGraph g; auto A=g.var(), B=g.var(), C=g.var();
  auto expr = g.mod(g.mul(A,B), C);                  // (A*B) % C  (no common factor guarantee)
  // stay as non-affine mod
  EXPECT_TRUE(g.resolve(expr).isSymbolic());
}

TEST(symbolic, unsolvable_mod_nested_symbolic_mixed) {
  vkcnn::SymGraph g; auto X=g.var(), A=g.var(), B=g.var();
  auto expr = g.mod(g.mod(X, A), B);                 // cannot combine without knowing gcd(A,B)
  EXPECT_TRUE(g.resolve(expr).isSymbolic());
}

TEST(symbolic, unsolvable_mod_nested_symbolic_must_not_drop_outer) {
  vkcnn::SymGraph g; auto X=g.var(), A=g.var(), B=g.var();
  auto inner = g.mod(X, A);
  auto expr  = g.mod(inner, B);      // (X % A) % B
  EXPECT_NE(inner, expr);            // must NOT simplify to X % A (unless B==A)
  auto same  = g.mod(inner, A);
  EXPECT_EQ(inner, same);            // idempotence only when A==A
}



