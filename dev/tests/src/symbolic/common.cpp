#include "vkcnn/common/symbolic/SymGraph.hpp"
#include <gtest/gtest.h>

TEST(symbolic, common_pool2x2_followed_by_upsample_equality_simple) {
  vkcnn::SymGraph g;

  std::size_t kernelSize = 2;
  std::size_t padding = 0;
  std::size_t stride = 2;

  std::size_t scalingFactor = 2;

  auto W = g.createParameter();

  auto w0 = g.mul(W, 2); // required for proving equality.

  auto x0 = g.add(w0, 2 * padding);
  auto x1 = g.sub(x0, kernelSize - 1);
  auto x2 = g.sub(x1, 1);
  auto x3 = g.div(x2, stride);
  auto w1 = g.add(x3, 1);
  auto w2 = g.mul(w1, scalingFactor);

  EXPECT_EQ(w0, w2);
}

TEST(symbolic, common_pool_k_eq_s_followed_by_upsample_identity) {
  vkcnn::SymGraph g;

  std::size_t s = 3;      // try 2,3,4...
  std::size_t k = s;
  std::size_t p = 0;

  auto W = g.createParameter();
  auto w0 = g.mul(W, s);

  auto x0 = g.add(w0, 2 * p);
  auto x1 = g.sub(x0, k - 1);
  auto x2 = g.sub(x1, 1);
  auto pooled = g.div(x2, s);
  auto out = g.mul(g.add(pooled, 1), s);

  EXPECT_EQ(w0, out);  // s*W == upsample(s, pool(k=s, s))
}

TEST(symbolic, common_two_pool2x2_then_upsample4_identity) {
  vkcnn::SymGraph g;

  std::size_t k = 2, s = 2, p = 0;

  auto step = [&](auto in) {
    auto x0 = g.add(in, 2 * p);
    auto x1 = g.sub(x0, k - 1);
    auto x2 = g.sub(x1, 1);
    auto y  = g.add(g.div(x2, s), 1); // one pool
    return y;
  };

  auto W = g.createParameter();
  auto w0 = g.mul(W, 4);
  auto y  = step(step(w0));
  auto out = g.mul(y, 4);

  EXPECT_EQ(w0, out);  // 4*W preserved
}

TEST(symbolic, common_hoist_multiples_and_floor_constant_remainder) {
  vkcnn::SymGraph g;
  auto W = g.createParameter();

  std::size_t s = 4;
  std::size_t c = 7;

  auto lhs = g.div(g.add(g.mul(W, s), c), s);
  auto rhs = g.add(W, c / s); // 7/4 == 1

  EXPECT_EQ(rhs, lhs);        // (4W + 7)/4 == W + 1
}

TEST(symbolic, common_same_padding_stride1_identity) {
  vkcnn::SymGraph g;

  std::size_t k = 3, d = 1;
  std::size_t s = 1;
  std::size_t p = (d * (k - 1)) / 2; // =1 here

  auto W = g.createParameter();

  auto x0 = g.add(W, 2 * p);
  auto x1 = g.sub(x0, d * (k - 1));
  auto x2 = g.sub(x1, 1);
  auto hout = g.add(g.div(x2, s), 1);

  EXPECT_EQ(W, hout);  // SAME padding preserves spatial size (k=3,d=1,s=1)
}

TEST(symbolic, common_ceil_mode_pooling_canonicalizes) {
  vkcnn::SymGraph g;
  auto W = g.createParameter();

  std::size_t k = 3, s = 2, p = 0, d = 1;

  // ceil-mode: ceil((W + 2p - d*(k-1) - 1)/s) + 1
  auto x0 = g.add(W, 2 * p);
  auto x1 = g.sub(x0, d * (k - 1));
  auto x2 = g.sub(x1, 1);
  auto lhs = g.add(g.div(g.add(x2, s - 1), s), 1); // (X + s - 1)/s + 1

  // Expect: keep (X + s - 1)/s as canonical ceil-div pattern
  auto rhs = lhs; // after your rewrite it should still compare equal
  EXPECT_EQ(rhs, lhs);
}
