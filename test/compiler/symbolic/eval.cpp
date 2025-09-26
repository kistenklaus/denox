#include "memory/container/vector.hpp"
#include "symbolic/SymGraph.hpp"
#include <gtest/gtest.h>

using namespace denox::compiler;
using namespace denox;

TEST(symbolic, eval_simple_add) {
  SymGraph symGraph;
  auto X = symGraph.var();
  
  auto xp1 = symGraph.add(X, 1);

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), 1);
  auto eval = symGraph.eval(specs);
  auto xp1_value = *eval[xp1];


  EXPECT_EQ(xp1_value, 1 + 1);
}


TEST(symbolic, eval_simple_sub) {
  SymGraph symGraph;
  auto X = symGraph.var();
  
  auto xp1 = symGraph.sub(X, 1);

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), 1);
  auto eval = symGraph.eval(specs);
  auto xp1_value = *eval[xp1];


  EXPECT_EQ(xp1_value, 1 - 1);
}

TEST(symbolic, eval_simple_mul) {
  SymGraph symGraph;
  auto X = symGraph.var();
  
  auto xp1 = symGraph.mul(X, 2);

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), 2);

  auto eval = symGraph.eval(specs);
  auto xp1_value = *eval[xp1];


  EXPECT_EQ(xp1_value, 2 * 2);
}

TEST(symbolic, eval_simple_div) {
  SymGraph symGraph;
  auto X = symGraph.var();
  
  auto xp1 = symGraph.div(X, 2);

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), 4);
  auto eval = symGraph.eval(specs);
  auto xp1_value = *eval[xp1];


  EXPECT_EQ(xp1_value, 4 / 2);
}


TEST(symbolic, eval_simple_mod) {
  SymGraph symGraph;
  auto X = symGraph.var();
  
  auto xp1 = symGraph.div(X, 2);

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), 3);
  auto eval = symGraph.eval(specs);
  auto xp1_value = *eval[xp1];


  EXPECT_EQ(xp1_value, 3 % 2);
}

TEST(symbolic, eval_add_mul_chain) {
  SymGraph symGraph;
  auto X = symGraph.var();

  auto xp1 = symGraph.add(X, 1);
  auto times3 = symGraph.mul(xp1, 3);

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), 2);

  auto eval = symGraph.eval(specs);
  auto v = *eval[times3];

  EXPECT_EQ(v, (2 + 1) * 3);
}

TEST(symbolic, eval_two_vars_sum_and_scale) {
  SymGraph symGraph;
  auto X = symGraph.var();
  auto Y = symGraph.var();

  auto sum = symGraph.add(X, Y);
  auto scaled = symGraph.mul(sum, 2);

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), 3);
  specs.emplace_back(Y.sym(), 4);

  auto eval = symGraph.eval(specs);
  EXPECT_EQ(*eval[sum], 3 + 4);
  EXPECT_EQ(*eval[scaled], (3 + 4) * 2);
}

TEST(symbolic, eval_associativity_add_and_mul) {
  SymGraph symGraph;
  auto X = symGraph.var();
  auto Y = symGraph.var();
  auto Z = symGraph.var();

  auto add1 = symGraph.add(symGraph.add(X, Y), Z);
  auto add2 = symGraph.add(X, symGraph.add(Y, Z));

  auto mul1 = symGraph.mul(symGraph.mul(X, Y), Z);
  auto mul2 = symGraph.mul(X, symGraph.mul(Y, Z));

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), 2);
  specs.emplace_back(Y.sym(), 3);
  specs.emplace_back(Z.sym(), 4);

  auto eval = symGraph.eval(specs);
  EXPECT_EQ(*eval[add1], *eval[add2]);
  EXPECT_EQ(*eval[mul1], *eval[mul2]);
  EXPECT_EQ(*eval[add1], 2 + 3 + 4);
  EXPECT_EQ(*eval[mul1], 2 * 3 * 4);
}

TEST(symbolic, eval_distributivity) {
  SymGraph symGraph;
  auto X = symGraph.var();
  auto Y = symGraph.var();
  auto Z = symGraph.var();

  auto left  = symGraph.mul(symGraph.add(X, Y), Z);
  auto right = symGraph.add(symGraph.mul(X, Z), symGraph.mul(Y, Z));

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), 5);
  specs.emplace_back(Y.sym(), 7);
  specs.emplace_back(Z.sym(), 3);

  auto eval = symGraph.eval(specs);
  EXPECT_EQ(*eval[left], *eval[right]);
  EXPECT_EQ(*eval[left], (5 + 7) * 3);
}

TEST(symbolic, eval_polynomial_xy) {
  // P(X,Y) = 3*X*X + 2*X*Y - 5*Y + 7
  SymGraph symGraph;
  auto X = symGraph.var();
  auto Y = symGraph.var();

  auto x2     = symGraph.mul(X, X);
  auto t1     = symGraph.mul(x2, 3);
  auto xy     = symGraph.mul(X, Y);
  auto t2     = symGraph.mul(xy, 2);
  auto t3     = symGraph.mul(Y, 5);
  auto acc1   = symGraph.add(t1, t2);
  auto acc2   = symGraph.sub(acc1, t3);
  auto poly   = symGraph.add(acc2, 7);

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), 3);
  specs.emplace_back(Y.sym(), 2);

  auto eval = symGraph.eval(specs);
  EXPECT_EQ(*eval[poly], 3*3*3 + 2*3*2 - 5*2 + 7); // 27 + 12 - 10 + 7 = 36
}

TEST(symbolic, eval_reuse_subgraph) {
  SymGraph symGraph;
  auto X = symGraph.var();
  auto Y = symGraph.var();

  auto A  = symGraph.add(X, Y);
  auto B  = symGraph.add(A, A); // 2*(X+Y)

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), 1);
  specs.emplace_back(Y.sym(), 2);

  auto eval = symGraph.eval(specs);
  EXPECT_EQ(*eval[B], 2 * (1 + 2));
}

TEST(symbolic, eval_specs_order_independence) {
  SymGraph symGraph;
  auto X = symGraph.var();
  auto Y = symGraph.var();

  auto sum = symGraph.add(X, Y);

  memory::vector<SymSpec> specs;
  // Intentionally add Y first, then X
  specs.emplace_back(Y.sym(), 20);
  specs.emplace_back(X.sym(), 10);

  auto eval = symGraph.eval(specs);
  EXPECT_EQ(*eval[sum], 30);
}

TEST(symbolic, eval_multiple_outputs_single_eval) {
  SymGraph symGraph;
  auto X = symGraph.var();
  auto Y = symGraph.var();

  auto sum   = symGraph.add(X, Y);
  auto thrice = symGraph.mul(sum, 3);

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), 2);
  specs.emplace_back(Y.sym(), 5);

  auto eval = symGraph.eval(specs);
  EXPECT_EQ(*eval[sum], 7);
  EXPECT_EQ(*eval[thrice], 21);
}

TEST(symbolic, eval_re_evaluate_with_different_specs) {
  SymGraph symGraph;
  auto X = symGraph.var();

  auto twice_plus_five = symGraph.add(symGraph.mul(X, 2), 5);

  memory::vector<SymSpec> s1;
  s1.emplace_back(X.sym(), 10);

  memory::vector<SymSpec> s2;
  s2.emplace_back(X.sym(), -3);

  auto e1 = symGraph.eval(s1);
  auto e2 = symGraph.eval(s2);

  EXPECT_EQ(*e1[twice_plus_five], 2*10 + 5);
  EXPECT_EQ(*e2[twice_plus_five], 2*(-3) + 5);
}

TEST(symbolic, eval_negative_arithmetic_mix) {
  SymGraph symGraph;
  auto X = symGraph.var();
  auto Y = symGraph.var();

  // E = X*Y + (Y/3) - (X - 2)
  auto xy   = symGraph.mul(X, Y);
  auto y3   = symGraph.div(Y, 3);
  auto xm2  = symGraph.sub(X, 2);
  auto part = symGraph.sub(y3, xm2);
  auto E    = symGraph.add(xy, part);

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), -3);
  specs.emplace_back(Y.sym(), 7);

  auto eval = symGraph.eval(specs);
  EXPECT_EQ(*eval[E], (-3)*7 + (7/3) - (-3 - 2)); // -21 + 2 - (-5) = -14
}

TEST(symbolic, eval_division_truncates_toward_zero_for_negatives) {
  SymGraph symGraph;
  auto X = symGraph.var();
  auto q = symGraph.div(X, 3);

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), -7);

  auto eval = symGraph.eval(specs);
  EXPECT_EQ(*eval[q], -7 / 3); // -2 in C++ (toward zero)
}

TEST(symbolic, eval_long_add_chain_depth_200) {
  SymGraph symGraph;
  auto X = symGraph.var();

  auto acc = X;
  for (int i = 0; i < 200; ++i) {
    acc = symGraph.add(acc, 1);
  }

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), 5);

  auto eval = symGraph.eval(specs);
  EXPECT_EQ(*eval[acc], 5 + 200);
}

TEST(symbolic, eval_commutativity_add_mul) {
  SymGraph symGraph;
  auto X = symGraph.var();
  auto Y = symGraph.var();

  auto sumXY = symGraph.add(X, Y);
  auto sumYX = symGraph.add(Y, X);

  auto mulXY = symGraph.mul(X, Y);
  auto mulYX = symGraph.mul(Y, X);

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), 8);
  specs.emplace_back(Y.sym(), 13);

  auto eval = symGraph.eval(specs);
  EXPECT_EQ(*eval[sumXY], *eval[sumYX]);
  EXPECT_EQ(*eval[mulXY], *eval[mulYX]);
  EXPECT_EQ(*eval[sumXY], 8 + 13);
  EXPECT_EQ(*eval[mulXY], 8 * 13);
}

TEST(symbolic, eval_identities_zero_one) {
  SymGraph symGraph;
  auto X = symGraph.var();

  auto plus0  = symGraph.add(X, 0);
  auto times1 = symGraph.mul(X, 1);
  auto minus0 = symGraph.sub(X, 0);
  auto div1   = symGraph.div(X, 1);

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), 9);

  auto eval = symGraph.eval(specs);
  EXPECT_EQ(*eval[plus0], 9);
  EXPECT_EQ(*eval[times1], 9);
  EXPECT_EQ(*eval[minus0], 9);
  EXPECT_EQ(*eval[div1], 9);
}

TEST(symbolic, eval_multiply_by_zero) {
  SymGraph symGraph;
  auto X = symGraph.var();
  auto zero = symGraph.mul(X, 0);

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), 17);

  auto eval = symGraph.eval(specs);
  EXPECT_EQ(*eval[zero], 0);
}

TEST(symbolic, eval_three_vars_mixed_expression) {
  // E = (X + Y + Z) * (Z - 2) + X*Z - Y
  SymGraph symGraph;
  auto X = symGraph.var();
  auto Y = symGraph.var();
  auto Z = symGraph.var();

  auto sum   = symGraph.add(symGraph.add(X, Y), Z);
  auto zm2   = symGraph.sub(Z, 2);
  auto part1 = symGraph.mul(sum, zm2);
  auto part2 = symGraph.mul(X, Z);
  auto tmp   = symGraph.add(part1, part2);
  auto E     = symGraph.sub(tmp, Y);

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), 2);
  specs.emplace_back(Y.sym(), 3);
  specs.emplace_back(Z.sym(), 4);

  auto eval = symGraph.eval(specs);
  EXPECT_EQ(*eval[E], (2+3+4)*(4-2) + 2*4 - 3); // 23
}

TEST(symbolic, eval_div_remainder_property_nonnegative) {
  // For nonnegative X and positive d: X == (X/d)*d + (X - (X/d)*d)
  SymGraph symGraph;
  auto X = symGraph.var();

  int d = 4;
  auto q    = symGraph.div(X, d);
  auto qd   = symGraph.mul(q, d);
  auto r    = symGraph.sub(X, qd);
  [[maybe_unused]] auto sum  = symGraph.add(qd, 0);  // just to use qd twice cleanly

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), 10);

  auto eval = symGraph.eval(specs);
  auto qv = *eval[q];
  auto qdv = *eval[qd];
  auto rv = *eval[r];

  EXPECT_EQ(qv, 10 / d);
  EXPECT_EQ(qdv + rv, 10);
  EXPECT_EQ(rv, 10 % d);
}

TEST(symbolic, eval_even_div_mul_inverse_when_even) {
  SymGraph symGraph;
  auto X = symGraph.var();

  auto half = symGraph.div(X, 2);
  auto twice_half = symGraph.mul(half, 2);

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), 14);

  auto eval = symGraph.eval(specs);
  EXPECT_EQ(*eval[twice_half], 14);
}

TEST(symbolic, eval_larger_mixed_expression) {
  // E = (X - Y) * (X + Y) + (X/2) * (Y/2) - 1
  SymGraph symGraph;
  auto X = symGraph.var();
  auto Y = symGraph.var();

  auto xmY   = symGraph.sub(X, Y);
  auto xpY   = symGraph.add(X, Y);
  auto prod1 = symGraph.mul(xmY, xpY);
  auto xh    = symGraph.div(X, 2);
  auto yh    = symGraph.div(Y, 2);
  auto prod2 = symGraph.mul(xh, yh);
  auto sum   = symGraph.add(prod1, prod2);
  auto E     = symGraph.sub(sum, 1);

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), 12);
  specs.emplace_back(Y.sym(), 8);

  auto eval = symGraph.eval(specs);
  EXPECT_EQ(*eval[E], (12-8)*(12+8) + (12/2)*(8/2) - 1); // 103
}

TEST(symbolic, eval_nested_constants_and_vars) {
  // ((X + 5) - 3) * ((Y * 2) / 3) + 7
  SymGraph symGraph;
  auto X = symGraph.var();
  auto Y = symGraph.var();

  auto xp5   = symGraph.add(X, 5);
  auto xp2   = symGraph.sub(xp5, 3);
  auto y2    = symGraph.mul(Y, 2);
  auto y2d3  = symGraph.div(y2, 3);
  auto prod  = symGraph.mul(xp2, y2d3);
  auto E     = symGraph.add(prod, 7);

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), 4);  // (4+5-3)=6
  specs.emplace_back(Y.sym(), 7);  // (7*2)/3=4

  auto eval = symGraph.eval(specs);
  EXPECT_EQ(*eval[E], 6 * 4 + 7); // 31
}

TEST(symbolic, eval_double_use_of_result_in_two_paths) {
  SymGraph symGraph;
  auto X = symGraph.var();
  auto Y = symGraph.var();

  auto S     = symGraph.add(X, Y);
  auto Sx2   = symGraph.mul(S, 2);
  auto Sx3   = symGraph.mul(S, 3);
  auto total = symGraph.add(Sx2, Sx3); // 5*(X+Y)

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), 6);
  specs.emplace_back(Y.sym(), 9);

  auto eval = symGraph.eval(specs);
  EXPECT_EQ(*eval[total], 5 * (6 + 9));
}

TEST(symbolic, eval_symmetric_build_paths_same_result) {
  SymGraph symGraph;
  auto X = symGraph.var();
  auto Y = symGraph.var();

  auto left  = symGraph.sub(symGraph.mul(symGraph.add(X, 1), Y), 2);
  auto right = symGraph.sub(symGraph.mul(Y, symGraph.add(X, 1)), 2);

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), 3);
  specs.emplace_back(Y.sym(), 5);

  auto eval = symGraph.eval(specs);
  EXPECT_EQ(*eval[left], *eval[right]);
  EXPECT_EQ(*eval[left], (3 + 1) * 5 - 2);
}
