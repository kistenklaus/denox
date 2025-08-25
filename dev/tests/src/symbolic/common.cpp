#include "vkcnn/common/symbolic/SymGraph.hpp"
#include <gtest/gtest.h>

TEST(symbolic, common_cnn_align_pool2x2_upsample2_single) {
  vkcnn::SymGraph g;
  auto W = g.var();

  auto W_aligned = g.alignUp(W, 2);

  auto W_post_pool = g.pool(W_aligned, 2, 0, 2);

  auto W_post_upsample = g.mul(W_post_pool, 2);

  EXPECT_EQ(W_aligned, W_post_upsample);
}


TEST(symbolic, common_cnn_align_pool2x2_upsample2_multiple_layers) {
  vkcnn::SymGraph g;
  auto W = g.var();

  auto W_aligned = g.alignUp(W, 16);

  auto W_post_pool0 = g.pool(W_aligned, 2, 0, 2);
  auto W_post_pool1 = g.pool(W_post_pool0, 2, 0, 2);
  auto W_post_pool2 = g.pool(W_post_pool1, 2, 0, 2);
  auto W_post_pool3 = g.pool(W_post_pool2, 2, 0, 2);

  auto W_post_upsample0 = g.mul(W_post_pool3, 2);
  auto W_post_upsample1 = g.mul(W_post_upsample0, 2);
  auto W_post_upsample2 = g.mul(W_post_upsample1, 2);
  auto W_post_upsample3 = g.mul(W_post_upsample2, 2);

  EXPECT_EQ(W_post_pool2, W_post_upsample0);
  EXPECT_EQ(W_post_pool1, W_post_upsample1);
  EXPECT_EQ(W_post_pool0, W_post_upsample2);
  EXPECT_EQ(W_aligned, W_post_upsample3);
}

TEST(symbolic, common_cnn_modalign_pool2x2_upsample2_single) {
  vkcnn::SymGraph g;
  auto W = g.var();

  std::size_t alignment = 2;
  auto W_aligned = g.add(W, g.sub(alignment, g.mod(W, alignment)));

  auto W_post_pool = g.pool(W_aligned, 2, 0, 2);

  auto W_post_upsample = g.mul(W_post_pool, 2);

  EXPECT_EQ(W_aligned, W_post_upsample);
}


// TEST(symbolic, common_cnn_modalign_pool2x2_upsample2_multiple_layers) {
//   vkcnn::SymGraph g;
//   auto W = g.var();
//
//   std::size_t alignment = 16;
//   auto W_aligned = g.add(W, g.sub(alignment, g.mod(W, alignment)));
//
//   auto W_post_pool0 = g.pool(W_aligned, 2, 0, 2);
//   auto W_post_pool1 = g.pool(W_post_pool0, 2, 0, 2);
//   auto W_post_pool2 = g.pool(W_post_pool1, 2, 0, 2);
//   auto W_post_pool3 = g.pool(W_post_pool2, 2, 0, 2);
//
//   auto W_post_upsample0 = g.mul(W_post_pool3, 2);
//   auto W_post_upsample1 = g.mul(W_post_upsample0, 2);
//   auto W_post_upsample2 = g.mul(W_post_upsample1, 2);
//   auto W_post_upsample3 = g.mul(W_post_upsample2, 2);
//
//   EXPECT_EQ(W_post_pool2, W_post_upsample0);
//   EXPECT_EQ(W_post_pool1, W_post_upsample1);
//   EXPECT_EQ(W_post_pool0, W_post_upsample2);
//   EXPECT_EQ(W_aligned, W_post_upsample3);
// }
