#include "vkcnn/common/symbolic/SymGraph.hpp"
#include <gtest/gtest.h>

TEST(symbolic, unsolvable_div_not_divisible_bails) {
  vkcnn::SymGraph g;
  auto A = g.createParameter();
  auto e = g.add(g.mul(6, A), 8);  // 6A+8
  auto q = g.resolve(g.div(e, 4)); // 6A/4 not integer → symbolic
  EXPECT_TRUE(q.isSymbolic());
}

TEST(symbolic, unsolvable_div_by_var_nontrivial) {
  vkcnn::SymGraph g;
  auto A = g.createParameter();
  // (2A + 1) / A  -> not a constant (your A/A special-case doesn't apply)
  auto q = g.resolve(g.div(g.add(g.mul(2, A), 1), A));
  EXPECT_TRUE(q.isSymbolic());
}

TEST(symbolic, unsolvable_div_const_not_all_divisible) {
  vkcnn::SymGraph g;
  auto A = g.createParameter();
  // (A + 1) / 2  -> some terms not divisible by 2 => bail
  auto q = g.resolve(g.div(g.add(A, 1), 2));
  EXPECT_TRUE(q.isSymbolic());
}

TEST(symbolic, unsolvable_mod_const_not_all_divisible) {
  vkcnn::SymGraph g;
  auto A = g.createParameter();
  // (A + 1) % 2  -> coef 1 not divisible by 2 => bail
  auto r = g.resolve(g.mod(g.add(A, 1), 2));
  EXPECT_TRUE(r.isSymbolic());
}

TEST(symbolic, unsolvable_const_mod_affine_bails) {
  vkcnn::SymGraph g;
  auto A = g.createParameter();
  // 7 % (A + 2)  -> rhs non-constant => bail in Z-affine
  auto r = g.resolve(g.mod(7, g.add(A, 2)));
  EXPECT_TRUE(r.isSymbolic());
}

TEST(symbolic, unsolvable_affine_mod_affine_different_base_bails) {
  vkcnn::SymGraph g;
  auto A = g.createParameter();
  // (A + 1) % (A + 3)  -> bases differ, gcd-rule doesn't apply => bail
  auto r = g.resolve(g.mod(g.add(A, 1), g.add(A, 3)));
  EXPECT_TRUE(r.isSymbolic());
}

TEST(symbolic, unsolvable_div_const_not_divisible_some_terms) {
  vkcnn::SymGraph g;
  auto A = g.createParameter();
  // (4A + 2) / 3  -> coefficients not all divisible by 3 => bail
  auto q = g.resolve(g.div(g.add(g.mul(4, A), 2), 3));
  EXPECT_TRUE(q.isSymbolic());
}

TEST(symbolic, unsolvable_compound_bail_div_then_mod) {
  vkcnn::SymGraph g;
  auto A = g.createParameter();
  // (A + 2)/2  already bails -> symbol; then %2 should also bail in Z-affine
  auto r = g.resolve(g.mod(g.div(g.add(A, 2), 2), 2));
  EXPECT_TRUE(r.isSymbolic());
}

TEST(symbolic, unsolvable_mod_same_base_nonzero_remainder_is_affine_not_const) {
  vkcnn::SymGraph g;
  auto A = g.createParameter();
  auto E = g.add(A, 1); // affine base
  // (3E) % (2E) = (3%2) * E = E  -> symbolic (non-constant) but simplified
  auto r = g.resolve(g.mod(g.mul(3, E), g.mul(2, E)));
  EXPECT_EQ(r, E);
  EXPECT_TRUE(r.isSymbolic());
}

TEST(symbolic, unsolvable_parity) {
  vkcnn::SymGraph g;
  auto A = g.createParameter();
  auto r = g.resolve(g.mod(g.add(A, 1), 2)); // depends on A parity
  EXPECT_TRUE(r.isSymbolic());
}
