#include "vkcnn/common/symbolic/SymGraph.hpp"
#include <gtest/gtest.h>

using namespace vkcnn;

TEST(symbolic, modsolver_canonicalize_mod_m_n) {
  vkcnn::SymGraph g;
  auto X = g.var();
  // a = (3(X mod 16) + 5) mod 4
  auto a = g.mod(g.add(g.mul(3, g.mod(X, 16)), 5), 4);
  // b = (3X + 5) mod 4
  auto b = g.mod(g.add(g.mul(3, X), 5), 4);
  // Because 16 % 4 == 0 => a == b.
  EXPECT_EQ(a, b);
}

TEST(symbolic, modsolver_div_lifting_div4_mod4) {
  vkcnn::SymGraph g;
  auto W = g.var();

  auto U = g.add(g.sub(W, g.mod(W, 16)), 14); // U ≡ 14 (mod 16)
  auto Z = g.div(U, 4);                       // floor((16k+14)/4) = 4k+3

  // Z mod 4 must be 3 (requires lifting to modulus 16 internally)
  auto Z_mod4 = g.resolve(g.mod(Z, 4));
  ASSERT_FALSE(Z_mod4.isSymbolic());
  EXPECT_EQ(3, Z_mod4.value());
}
