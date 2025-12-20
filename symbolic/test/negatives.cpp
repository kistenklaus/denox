#include "memory/container/vector.hpp"
#include "symbolic/SymGraph.hpp"
#include <gtest/gtest.h>

using namespace denox::compiler;
using namespace denox;

// NOTE: A collection of tests that fail because we assume all values are
// positve! Not part of the test suite, but maybe if at some point we want to 
// also support negative numbers then this is a good starting point

TEST(symbolic, eval_division_sign_cases) {
  SymGraph g;
  auto X = g.var();

  // q1 = (-7)/3, q2 = 7/(-3), q3 = (-7)/(-3)
  auto q1 = g.div(g.add(0, g.mul(-1, X)), 3); // -X / 3
  auto q2 = g.div(X, -3);
  auto q3 = g.div(g.mul(-1, X), -3);

  memory::vector<SymSpec> s;
  s.emplace_back(X.sym(), 7);

  auto ev = g.eval(s);
  EXPECT_EQ(*ev[q1], -7 / 3);  // -2
  EXPECT_EQ(*ev[q2], 7 / -3);  // -2
  EXPECT_EQ(*ev[q3], -7 / -3); //  2
}

TEST(symbolic, eval_division_sign_cases_fixed) {
  SymGraph g;
  Sym X = g.var();

  Sym q1 = g.div(g.mul(X, -1), 3); // -X / 3

  g.debugDump();

  memory::vector<SymSpec> s;
  s.emplace_back(X.sym(), 7);

  auto ev = g.eval(s);
  EXPECT_EQ(*ev[q1], -7 / 3);  // -2
}

TEST(symbolic, division_negation_forms_equivalent) {
  SymGraph g;
  Sym X = g.var();

  Sym q_mul = g.div(g.mul(X, -1), 3);            // (-X)/3
  Sym q_add0 = g.div(g.add(0, g.mul(-1, X)), 3); // (-X)/3 via add(int, Sym)
  Sym q_sub0 = g.div(g.sub(0, X), 3);            // (-X)/3 via sub(0, X)

  memory::vector<SymSpec> s;
  s.emplace_back(X.sym(), 7);
  auto ev = g.eval(s);

  int expected = (-7) / 3; // -2 (C++ truncates toward zero)
  EXPECT_EQ(*ev[q_mul], expected);
  EXPECT_EQ(*ev[q_sub0], expected);
  EXPECT_EQ(*ev[q_add0], expected); // this is the one that currently gives -3
}

TEST(symbolic, add_left_zero_identity) {
  SymGraph g;
  Sym X = g.var();
  Sym a = g.add(X, 0);
  Sym b = g.add(0, X); // constant-first path again
  memory::vector<SymSpec> s;
  s.emplace_back(X.sym(), -9);
  auto ev = g.eval(s);
  EXPECT_EQ(*ev[a], -9);
  EXPECT_EQ(*ev[b],
            -9); // if this fails, add(int, Sym) is broken even without negation
}
