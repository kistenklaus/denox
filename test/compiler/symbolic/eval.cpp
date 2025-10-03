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

  auto left = symGraph.mul(symGraph.add(X, Y), Z);
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

  auto x2 = symGraph.mul(X, X);
  auto t1 = symGraph.mul(x2, 3);
  auto xy = symGraph.mul(X, Y);
  auto t2 = symGraph.mul(xy, 2);
  auto t3 = symGraph.mul(Y, 5);
  auto acc1 = symGraph.add(t1, t2);
  auto acc2 = symGraph.sub(acc1, t3);
  auto poly = symGraph.add(acc2, 7);

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), 3);
  specs.emplace_back(Y.sym(), 2);

  auto eval = symGraph.eval(specs);
  EXPECT_EQ(*eval[poly],
            3 * 3 * 3 + 2 * 3 * 2 - 5 * 2 + 7); // 27 + 12 - 10 + 7 = 36
}

TEST(symbolic, eval_reuse_subgraph) {
  SymGraph symGraph;
  auto X = symGraph.var();
  auto Y = symGraph.var();

  auto A = symGraph.add(X, Y);
  auto B = symGraph.add(A, A); // 2*(X+Y)

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

  auto sum = symGraph.add(X, Y);
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

  EXPECT_EQ(*e1[twice_plus_five], 2 * 10 + 5);
  EXPECT_EQ(*e2[twice_plus_five], 2 * (-3) + 5);
}

TEST(symbolic, eval_negative_arithmetic_mix) {
  SymGraph symGraph;
  auto X = symGraph.var();
  auto Y = symGraph.var();

  // E = X*Y + (Y/3) - (X - 2)
  auto xy = symGraph.mul(X, Y);
  auto y3 = symGraph.div(Y, 3);
  auto xm2 = symGraph.sub(X, 2);
  auto part = symGraph.sub(y3, xm2);
  auto E = symGraph.add(xy, part);

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), -3);
  specs.emplace_back(Y.sym(), 7);

  auto eval = symGraph.eval(specs);
  EXPECT_EQ(*eval[E], (-3) * 7 + (7 / 3) - (-3 - 2)); // -21 + 2 - (-5) = -14
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

  auto plus0 = symGraph.add(X, 0);
  auto times1 = symGraph.mul(X, 1);
  auto minus0 = symGraph.sub(X, 0);
  auto div1 = symGraph.div(X, 1);

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

  auto sum = symGraph.add(symGraph.add(X, Y), Z);
  auto zm2 = symGraph.sub(Z, 2);
  auto part1 = symGraph.mul(sum, zm2);
  auto part2 = symGraph.mul(X, Z);
  auto tmp = symGraph.add(part1, part2);
  auto E = symGraph.sub(tmp, Y);

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), 2);
  specs.emplace_back(Y.sym(), 3);
  specs.emplace_back(Z.sym(), 4);

  auto eval = symGraph.eval(specs);
  EXPECT_EQ(*eval[E], (2 + 3 + 4) * (4 - 2) + 2 * 4 - 3); // 23
}

TEST(symbolic, eval_div_remainder_property_nonnegative) {
  // For nonnegative X and positive d: X == (X/d)*d + (X - (X/d)*d)
  SymGraph symGraph;
  auto X = symGraph.var();

  int d = 4;
  auto q = symGraph.div(X, d);
  auto qd = symGraph.mul(q, d);
  auto r = symGraph.sub(X, qd);
  [[maybe_unused]] auto sum =
      symGraph.add(qd, 0); // just to use qd twice cleanly

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

  auto xmY = symGraph.sub(X, Y);
  auto xpY = symGraph.add(X, Y);
  auto prod1 = symGraph.mul(xmY, xpY);
  auto xh = symGraph.div(X, 2);
  auto yh = symGraph.div(Y, 2);
  auto prod2 = symGraph.mul(xh, yh);
  auto sum = symGraph.add(prod1, prod2);
  auto E = symGraph.sub(sum, 1);

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), 12);
  specs.emplace_back(Y.sym(), 8);

  auto eval = symGraph.eval(specs);
  EXPECT_EQ(*eval[E], (12 - 8) * (12 + 8) + (12 / 2) * (8 / 2) - 1); // 103
}

TEST(symbolic, eval_nested_constants_and_vars) {
  // ((X + 5) - 3) * ((Y * 2) / 3) + 7
  SymGraph symGraph;
  auto X = symGraph.var();
  auto Y = symGraph.var();

  auto xp5 = symGraph.add(X, 5);
  auto xp2 = symGraph.sub(xp5, 3);
  auto y2 = symGraph.mul(Y, 2);
  auto y2d3 = symGraph.div(y2, 3);
  auto prod = symGraph.mul(xp2, y2d3);
  auto E = symGraph.add(prod, 7);

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), 4); // (4+5-3)=6
  specs.emplace_back(Y.sym(), 7); // (7*2)/3=4

  auto eval = symGraph.eval(specs);
  EXPECT_EQ(*eval[E], 6 * 4 + 7); // 31
}

TEST(symbolic, eval_double_use_of_result_in_two_paths) {
  SymGraph symGraph;
  auto X = symGraph.var();
  auto Y = symGraph.var();

  auto S = symGraph.add(X, Y);
  auto Sx2 = symGraph.mul(S, 2);
  auto Sx3 = symGraph.mul(S, 3);
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

  auto left = symGraph.sub(symGraph.mul(symGraph.add(X, 1), Y), 2);
  auto right = symGraph.sub(symGraph.mul(Y, symGraph.add(X, 1)), 2);

  memory::vector<SymSpec> specs;
  specs.emplace_back(X.sym(), 3);
  specs.emplace_back(Y.sym(), 5);

  auto eval = symGraph.eval(specs);
  EXPECT_EQ(*eval[left], *eval[right]);
  EXPECT_EQ(*eval[left], (3 + 1) * 5 - 2);
}

TEST(symbolic, eval_dot_product_100_vars) {
  SymGraph g;
  std::vector<Sym> V;
  V.reserve(100);
  for (int i = 0; i < 100; ++i)
    V.push_back(g.var());

  // sum_{i=1..100} i * V[i-1]
  auto acc = g.mul(V[0], 1);
  for (int i = 2; i <= 100; ++i) {
    acc = g.add(acc, g.mul(V[static_cast<std::size_t>(i - 1)], i));
  }

  memory::vector<SymSpec> specs;
  specs.reserve(100);
  for (int i = 1; i <= 100; ++i)
    specs.emplace_back(V[static_cast<std::size_t>(i - 1)].sym(), i);

  auto ev = g.eval(specs);

  // expected = sum i*i
  int expected = 0;
  for (int i = 1; i <= 100; ++i)
    expected += i * i;

  EXPECT_EQ(*ev[acc], expected);
}

TEST(symbolic, eval_balanced_vs_unbalanced_sum_64_vars) {
  SymGraph g;
  std::vector<Sym> V(64);
  for (int i = 0; i < 64; ++i)
    V[static_cast<std::size_t>(i)] = g.var();

  // Unbalanced left-fold: (((V0 + V1) + V2) + ... + V63)
  auto unbalanced = V[0];
  for (int i = 1; i < 64; ++i)
    unbalanced = g.add(unbalanced, V[static_cast<std::size_t>(i)]);

  // Balanced tree: pairwise adds until one remains
  std::vector<Sym> layer = V;
  while (layer.size() > 1) {
    std::vector<Sym> next;
    next.reserve((layer.size() + 1) / 2);
    for (size_t i = 0; i + 1 < layer.size(); i += 2) {
      next.push_back(g.add(layer[i], layer[i + 1]));
    }
    if (layer.size() % 2 == 1)
      next.push_back(layer.back());
    layer.swap(next);
  }
  auto balanced = layer[0];

  memory::vector<SymSpec> specs;
  specs.reserve(64);
  for (int i = 0; i < 64; ++i)
    specs.emplace_back(V[static_cast<std::size_t>(i)].sym(), i + 1);

  auto ev = g.eval(specs);
  int expected = (64 * (64 + 1)) / 2; // 1..64
  EXPECT_EQ(*ev[unbalanced], expected);
  EXPECT_EQ(*ev[balanced], expected);
}

TEST(symbolic, eval_matrix2x2_multiply) {
  SymGraph g;
  // A = [a b; c d], B = [e f; g h]
  auto a = g.var(), b = g.var(), c = g.var(), d = g.var();
  auto e = g.var(), f = g.var(), g2 = g.var(), h = g.var();

  // M = A * B
  auto m00 = g.add(g.mul(a, e), g.mul(b, g2));
  auto m01 = g.add(g.mul(a, f), g.mul(b, h));
  auto m10 = g.add(g.mul(c, e), g.mul(d, g2));
  auto m11 = g.add(g.mul(c, f), g.mul(d, h));

  memory::vector<SymSpec> s = {{a.sym(), 1},  {b.sym(), 2}, {c.sym(), 3},
                               {d.sym(), 4},  {e.sym(), 5}, {f.sym(), 6},
                               {g2.sym(), 7}, {h.sym(), 8}};

  auto ev = g.eval(s);
  EXPECT_EQ(*ev[m00], 1 * 5 + 2 * 7);
  EXPECT_EQ(*ev[m01], 1 * 6 + 2 * 8);
  EXPECT_EQ(*ev[m10], 3 * 5 + 4 * 7);
  EXPECT_EQ(*ev[m11], 3 * 6 + 4 * 8);
}

TEST(symbolic, eval_fibonacci_like_dag_F20) {
  // F0=0, F1=1, Fn=Fn-1+Fn-2 using graph adds
  SymGraph g;
  // Use constants by mul(X,0)+const; but simpler: maintain separate stream via
  // integers We'll seed with two const nodes: c0=0, c1=1 using var pinned specs
  auto F0 = g.var();
  auto F1 = g.var();

  std::vector<decltype(F0)> F(21);
  F[0] = F0;
  F[1] = F1;
  for (int n = 2; n <= 20; ++n)
    F[static_cast<std::size_t>(n)] = g.add(F[static_cast<std::size_t>(n - 1)],
                                           F[static_cast<std::size_t>(n - 2)]);

  memory::vector<SymSpec> s;
  s.emplace_back(F0.sym(), 0);
  s.emplace_back(F1.sym(), 1);
  auto ev = g.eval(s);

  // Fib(20)=6765
  EXPECT_EQ(*ev[F[20]], 6765);
}

TEST(symbolic, eval_diamond_sharing_stress_k50) {
  SymGraph g;
  auto X = g.var();
  auto Y = g.var();

  auto base = g.add(X, Y);

  // sum_{i=1..50} base  => 50*(X+Y)
  auto acc = base;
  for (int i = 1; i < 50; ++i)
    acc = g.add(acc, base);

  auto kxy = g.mul(g.add(X, Y), 50);

  memory::vector<SymSpec> s;
  s.emplace_back(X.sym(), 7);
  s.emplace_back(Y.sym(), -4);

  auto ev = g.eval(s);
  EXPECT_EQ(*ev[acc], *ev[kxy]);
  EXPECT_EQ(*ev[acc], 50 * (7 + (-4)));
}

TEST(symbolic, eval_many_outputs_single_eval_40_expressions) {
  SymGraph g;
  auto A = g.var();
  auto B = g.var();

  std::vector<decltype(A)> outs;
  outs.reserve(40);

  // Generate a zoo of expressions reusing subgraphs
  auto S = g.add(A, B);
  auto D = g.sub(A, B);
  auto P = g.mul(A, B);

  outs.push_back(S);
  outs.push_back(D);
  outs.push_back(P);

  // Chains and mixes
  auto t = A;
  for (int i = 0; i < 10; ++i) {
    t = g.add(t, i);
    outs.push_back(t);
  }
  auto u = B;
  for (int i = 1; i <= 10; ++i) {
    u = g.mul(u, i);
    outs.push_back(u);
  }
  auto v = g.add(g.mul(S, 3), g.sub(P, 2));
  outs.push_back(v);

  // A few more with divisions (no zeros)
  outs.push_back(g.div(g.add(P, 9), 3)); // (A*B+9)/3
  outs.push_back(g.add(g.div(S, 2), 1)); // S/2 + 1
  outs.push_back(g.sub(g.mul(D, D), 5)); // (A-B)^2 - 5

  memory::vector<SymSpec> s;
  s.emplace_back(A.sym(), 6);
  s.emplace_back(B.sym(), 3);

  auto ev = g.eval(s);

  // Spot checks
  EXPECT_EQ(*ev[S], 9);
  EXPECT_EQ(*ev[D], 3);
  EXPECT_EQ(*ev[P], 18);
  EXPECT_EQ(*ev[v], 3 * (6 + 3) + (18 - 2));           // 43
  EXPECT_EQ(*ev[outs[outs.size() - 3]], (18 + 9) / 3); // 9
  EXPECT_EQ(*ev[outs[outs.size() - 2]], (9 / 2) + 1);  // 5 (int div)
  EXPECT_EQ(*ev[outs.back()], (6 - 3) * (6 - 3) - 5);  // 4
}

TEST(symbolic, eval_deep_add_chain_1000) {
  SymGraph g;
  auto X = g.var();

  auto acc = X;
  for (int i = 0; i < 1000; ++i)
    acc = g.add(acc, 1);

  memory::vector<SymSpec> s;
  s.emplace_back(X.sym(), 123);

  auto ev = g.eval(s);
  EXPECT_EQ(*ev[acc], 1123);
}

TEST(symbolic, eval_alternating_add_sub_chain) {
  SymGraph g;
  auto X = g.var();

  auto acc = X;
  int expected = 5;
  for (int i = 1; i <= 200; ++i) {
    if (i % 2) {
      acc = g.add(acc, i);
      expected += i;
    } else {
      acc = g.sub(acc, i);
      expected -= i;
    }
  }

  memory::vector<SymSpec> s;
  s.emplace_back(X.sym(), 5);

  auto ev = g.eval(s);
  EXPECT_EQ(*ev[acc], expected);
}

TEST(symbolic, eval_identity_x_minus_x_zero_x_div_x_one_nonzero) {
  SymGraph g;
  auto X = g.var();

  auto zero = g.sub(X, X);
  auto one = g.div(X, X); // only valid for X!=0 in our spec

  memory::vector<SymSpec> s;
  s.emplace_back(X.sym(), 11);

  auto ev = g.eval(s);
  EXPECT_EQ(*ev[zero], 0);
  EXPECT_EQ(*ev[one], 1);
}

TEST(symbolic, eval_big_mixed_5_vars) {
  // E = ((A + B) * (C - D) + E) * ((A - E) + (C * D) - (B / 2))
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var(), D = g.var(), E = g.var();

  auto sumAB = g.add(A, B);
  auto CmD = g.sub(C, D);
  auto left = g.add(g.mul(sumAB, CmD), E);

  auto AmE = g.sub(A, E);
  auto CD = g.mul(C, D);
  auto Br2 = g.div(B, 2);
  auto right = g.sub(g.add(AmE, CD), Br2);

  auto expr = g.mul(left, right);

  memory::vector<SymSpec> s = {
      {A.sym(), 9}, {B.sym(), 5}, {C.sym(), 4}, {D.sym(), 3}, {E.sym(), -2}};

  auto ev = g.eval(s);

  int A_ = 9, B_ = 5, C_ = 4, D_ = 3, E_ = -2;
  int leftExp = (A_ + B_) * (C_ - D_) + E_;        // (14*1) -2 = 12
  int rightExp = (A_ - E_) + (C_ * D_) - (B_ / 2); // 11 + 12 - 2 = 21
  EXPECT_EQ(*ev[expr], leftExp * rightExp);        // 252
}

TEST(symbolic, eval_compare_two_factorizations) {
  // (X+Y)*(X-Y)  == X*X - Y*Y
  SymGraph g;
  auto X = g.var(), Y = g.var();

  auto left = g.mul(g.add(X, Y), g.sub(X, Y));
  auto right = g.sub(g.mul(X, X), g.mul(Y, Y));

  memory::vector<SymSpec> s = {{X.sym(), 13}, {Y.sym(), 7}};
  auto ev = g.eval(s);

  EXPECT_EQ(*ev[left], *ev[right]);
}

TEST(symbolic, eval_many_divisors_remainder_property_positive) {
  // For positive X and d>0: X == (X/d)*d + (X - (X/d)*d), repeated for d=2..9
  SymGraph g;
  auto X = g.var();

  memory::vector<SymSpec> s;
  s.emplace_back(X.sym(), 123);
  auto ev = g.eval(s);

  for (int d = 2; d <= 9; ++d) {
    auto q = g.div(X, d);
    auto qd = g.mul(q, d);
    auto r = g.sub(X, qd);

    auto ev2 = g.eval(s);
    Sym::value_type qv = *ev2[q];
    Sym::value_type qdv = *ev2[qd];
    Sym::value_type rv = *ev2[r];

    EXPECT_EQ(qv, 123 / d);
    EXPECT_EQ(qdv + rv, 123);
  }
}

TEST(symbolic, eval_large_linear_form_200_vars) {
  SymGraph g;
  std::vector<Sym> V(200);
  for (int i = 0; i < 200; ++i)
    V[static_cast<std::size_t>(i)] = g.var();

  // L = sum_{i=0..199} (i-50) * V[i]
  auto acc = g.mul(V[0], -50);
  for (int i = 1; i < 200; ++i)
    acc = g.add(acc, g.mul(V[static_cast<std::size_t>(i)], i - 50));

  memory::vector<SymSpec> s;
  s.reserve(200);
  for (int i = 0; i < 200; ++i)
    s.emplace_back(V[static_cast<std::size_t>(i)].sym(),
                   (i % 7) - 3); // cycle -3..3

  auto ev = g.eval(s);

  // expected in C++
  int expected = 0;
  for (int i = 0; i < 200; ++i)
    expected += (i - 50) * ((i % 7) - 3);

  EXPECT_EQ(*ev[acc], expected);
}

static void check_values(SymGraph &g, const Sym &a, const Sym &b, const Sym &X,
                         std::initializer_list<int> xs,
                         const std::function<int(int)> &expected) {
  for (int xv : xs) {
    memory::vector<SymSpec> s;
    s.emplace_back(X.sym(), xv);
    auto ev = g.eval(s);
    EXPECT_EQ(*ev[a], expected(xv));
    EXPECT_EQ(*ev[b], expected(xv));
  }
}

// Deg-2: P(x) = 7 + (-5)x + 3x^2
TEST(symbolic, eval_quad_horner_vs_naive_nodefirst) {
  SymGraph g;
  Sym X = g.var();

  // naive: 3x^2 + (-5)x + 7
  Sym x2 = g.mul(X, X);
  Sym accN = g.mul(x2, 3);
  accN = g.add(accN, g.mul(X, -5));
  accN = g.add(accN, 7);

  // Horner: (3x + (-5))x + 7
  Sym accH = g.mul(X, 3);
  accH = g.add(accH, -5);
  accH = g.mul(accH, X);
  accH = g.add(accH, 7);

  check_values(g, accN, accH, X, {-3, -1, 0, 1, 2, 5},
               [](int x) { return 7 + (-5) * x + 3 * x * x; });
}

// Deg-3 with zero x^2 term: P(x) = 7 + (-5)x + 0*x^2 + 3x^3
TEST(symbolic, eval_cubic_zero_middle_term_nodefirst) {
  SymGraph g;
  Sym X = g.var();

  Sym x2 = g.mul(X, X);
  Sym x3 = g.mul(x2, X);

  // naive: 3x^3 + 0*x^2 + (-5)x + 7
  Sym accN = g.mul(x3, 3);
  accN = g.add(accN, g.mul(x2, 0));
  accN = g.add(accN, g.mul(X, -5));
  accN = g.add(accN, 7);

  // Horner: ((3x + 0) x + (-5)) x + 7
  Sym accH = g.mul(X, 3);
  accH = g.add(accH, 0);
  accH = g.mul(accH, X);
  accH = g.add(accH, -5);
  accH = g.mul(accH, X);
  accH = g.add(accH, 7);

  check_values(g, accN, accH, X, {-3, -1, 0, 1, 2, 3},
               [](int x) { return 7 + (-5) * x + 0 * x * x + 3 * x * x * x; });
}

// Deg-3 with zero constant: P(x) = (-5)x + 2x^2 + 3x^3
TEST(symbolic, eval_cubic_zero_constant_nodefirst) {
  SymGraph g;
  Sym X = g.var();

  Sym x2 = g.mul(X, X);
  Sym x3 = g.mul(x2, X);

  // naive: 3x^3 + 2x^2 + (-5)x
  Sym accN = g.mul(x3, 3);
  accN = g.add(accN, g.mul(x2, 2));
  accN = g.add(accN, g.mul(X, -5));

  // Horner: ((3x + 2) x + (-5)) x
  Sym accH = g.mul(X, 3);
  accH = g.add(accH, 2);
  accH = g.mul(accH, X);
  accH = g.add(accH, -5);
  accH = g.mul(accH, X);

  check_values(g, accN, accH, X, {-4, -1, 0, 1, 2, 4},
               [](int x) { return (-5) * x + 2 * x * x + 3 * x * x * x; });
}

// Deg-4 like your failing pattern (explicit 0*x^3): P(x) = 7 + (-5)x + 2x^2 +
// 0*x^3 + 3x^4
TEST(symbolic, eval_quartic_zero_cubic_term_nodefirst) {
  SymGraph g;
  Sym X = g.var();

  Sym x2 = g.mul(X, X);
  Sym x3 = g.mul(x2, X);
  Sym x4 = g.mul(x3, X);

  // naive: 3x^4 + 0*x^3 + 2x^2 + (-5)x + 7
  Sym accN = g.mul(x4, 3);
  accN = g.add(accN, g.mul(x3, 0));
  accN = g.add(accN, g.mul(x2, 2));
  accN = g.add(accN, g.mul(X, -5));
  accN = g.add(accN, 7);

  // Horner: (((3x + 0) x + 2) x + (-5)) x + 7
  Sym accH = g.mul(X, 3);
  accH = g.add(accH, 0);
  accH = g.mul(accH, X);
  accH = g.add(accH, 2);
  accH = g.mul(accH, X);
  accH = g.add(accH, -5);
  accH = g.mul(accH, X);
  accH = g.add(accH, 7);

  check_values(g, accN, accH, X, {-3, -1, 0, 1, 2, 3, 5}, [](int x) {
    return 7 + (-5) * x + 2 * x * x + 0 * x * x * x + 3 * x * x * x * x;
  });
}

// Deg-4 with zero quadratic term: P(x) = 1 + (-4)x + 0*x^2 + 2x^3 + 5x^4
TEST(symbolic, eval_quartic_zero_quadratic_term_nodefirst) {
  SymGraph g;
  Sym X = g.var();

  Sym x2 = g.mul(X, X);
  Sym x3 = g.mul(x2, X);
  Sym x4 = g.mul(x3, X);

  // naive: 5x^4 + 2x^3 + 0*x^2 + (-4)x + 1
  Sym accN = g.mul(x4, 5);
  accN = g.add(accN, g.mul(x3, 2));
  accN = g.add(accN, g.mul(x2, 0));
  accN = g.add(accN, g.mul(X, -4));
  accN = g.add(accN, 1);

  // Horner: ((((5x + 2) x + 0) x + (-4)) x + 1)
  Sym accH = g.mul(X, 5);
  accH = g.add(accH, 2);
  accH = g.mul(accH, X);
  accH = g.add(accH, 0);
  accH = g.mul(accH, X);
  accH = g.add(accH, -4);
  accH = g.mul(accH, X);
  accH = g.add(accH, 1);

  check_values(g, accN, accH, X, {-2, -1, 0, 1, 2, 4}, [](int x) {
    return 1 + (-4) * x + 0 * x * x + 2 * x * x * x + 5 * x * x * x * x;
  });
}

// Deg-4 all negative coeffs (no zeros): P(x) = -7 + (-2)x + (-3)x^2 + (-4)x^3 +
// (-1)x^4
TEST(symbolic, eval_quartic_all_negative_nodefirst) {
  SymGraph g;
  Sym X = g.var();

  Sym x2 = g.mul(X, X);
  Sym x3 = g.mul(x2, X);
  Sym x4 = g.mul(x3, X);

  // naive: (-1)x^4 + (-4)x^3 + (-3)x^2 + (-2)x + (-7)
  Sym accN = g.mul(x4, -1);
  accN = g.add(accN, g.mul(x3, -4));
  accN = g.add(accN, g.mul(x2, -3));
  accN = g.add(accN, g.mul(X, -2));
  accN = g.add(accN, -7);

  // Horner: ((((-1)x + (-4)) x + (-3)) x + (-2)) x + (-7)
  Sym accH = g.mul(X, -1);
  accH = g.add(accH, -4);
  accH = g.mul(accH, X);
  accH = g.add(accH, -3);
  accH = g.mul(accH, X);
  accH = g.add(accH, -2);
  accH = g.mul(accH, X);
  accH = g.add(accH, -7);

  check_values(g, accN, accH, X, {-2, -1, 0, 1, 2}, [](int x) {
    return -7 + (-2) * x + (-3) * x * x + (-4) * x * x * x +
           (-1) * x * x * x * x;
  });
}

// Cross-check via expansion vs polynomial form: (x + 2)(x + 3) == x^2 + 5x + 6
TEST(symbolic, eval_binomial_expand_vs_poly_nodefirst) {
  SymGraph g;
  Sym X = g.var();

  Sym left = g.mul(g.add(X, 2), g.add(X, 3)); // (x+2)(x+3)
  Sym x2 = g.mul(X, X);
  Sym right = g.add(x2, g.add(g.mul(X, 5), 6)); // x^2 + 5x + 6

  check_values(g, left, right, X, {-3, -1, 0, 1, 2, 4},
               [](int x) { return (x + 2) * (x + 3); });
}

// Template deg-2 identity: ax^2 + bx + c == ((a x + b) x + c)
TEST(symbolic, eval_quad_template_identity_nodefirst) {
  SymGraph g;
  Sym X = g.var();

  // a=2, b=-7, c=5
  Sym x2 = g.mul(X, X);
  Sym poly = g.add(g.mul(x2, 2), g.add(g.mul(X, -7), 5));

  Sym h = g.mul(X, 2);
  h = g.add(h, -7);
  h = g.mul(h, X);
  h = g.add(h, 5);

  check_values(g, poly, h, X, {-4, -2, -1, 0, 1, 3},
               [](int x) { return 2 * x * x + (-7) * x + 5; });
}



