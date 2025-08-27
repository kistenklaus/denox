#include "vkcnn/common/symbolic/SymGraph.hpp"
#include <gtest/gtest.h>

using namespace vkcnn;

// ---------- Helpers: normalize-then-compare & idempotence ----------
#define NF(g, e) (g).resolve((e))
#define EQ_NF(g, a, b) EXPECT_EQ(NF(g, a), NF(g, b))

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

TEST(symbolic, common_cnn_modalign_pool2x2_upsample2_multiple_layers) {
  vkcnn::SymGraph g;
  auto W = g.var();

  std::size_t alignment = 16;
  auto W_aligned = g.add(W, g.sub(alignment, g.mod(W, alignment)));

  auto W_post_pool0 = g.pool(W_aligned, 2, 0, 2);
  auto W_post_pool1 = g.pool(W_post_pool0, 2, 0, 2);
  auto W_post_pool2 = g.pool(W_post_pool1, 2, 0, 2);
  auto W_post_pool3 = g.pool(W_post_pool2, 2, 0, 2);

  auto W_post_upsample0 = g.mul(W_post_pool3, 2);
  auto W_post_upsample1 = g.mul(W_post_upsample0, 2);
  auto W_post_upsample2 = g.mul(W_post_upsample1, 2);
  auto W_post_upsample3 = g.mul(W_post_upsample2, 2);

  EXPECT_EQ(W_post_pool2.sym(), W_post_upsample0.sym());
  EXPECT_EQ(W_post_pool1.sym(), W_post_upsample1.sym());
  EXPECT_EQ(W_post_pool0.sym(), W_post_upsample2.sym());
  EXPECT_EQ(W_aligned.sym(), W_post_upsample3.sym());
}

TEST(symbolic, common_cnn_modalign_pool2x2_upsample2_gigantic) {
  vkcnn::SymGraph g;
  auto W = g.var();

  std::uint64_t N = 60;
  // Image aligned to 1152921504606846976 xD, yeah that seems likely
  std::uint64_t alignment = std::uint64_t(1ULL) << N;
  auto W_aligned = g.add(W, g.sub(alignment, g.mod(W, alignment)));

  auto inout = W_aligned;
  for (std::size_t n = 0; n < N; ++n) {
    inout = g.pool(inout, 2, 0, 2);
  }
  for (std::size_t n = 0; n < N; ++n) {
    inout = g.mul(inout, 2);
  }
  EXPECT_EQ(inout.sym(), W_aligned.sym());
}

// -----------------------------------------------------------------------------
// 2) Zero-copy concat along width when K is a multiple of A:
//    If K := C*B is a multiple of A, every row size is already A-aligned.
//    Then concatenating widths is just addition (no extra padding needed):
//    alignUp(W1*K, A) + alignUp(W2*K, A) == alignUp((W1+W2)*K, A)
// -----------------------------------------------------------------------------
TEST(symbolic, common_concat_row_pitch_sum_equals_total_when_K_multiple_of_A) {
  vkcnn::SymGraph g;
  auto W1 = g.var(), W2 = g.var();
  // Example: C=16, B=2 (fp16), A=16  => K=32 is multiple of A
  int C = 16, B = 2, A = 16, K = C * B;
  auto rp1 = g.alignUp(g.mul(W1, K), A);
  auto rp2 = g.alignUp(g.mul(W2, K), A);
  auto rpsum = g.add(rp1, rp2);
  auto rptotal = g.alignUp(g.mul(g.add(W1, W2), K), A);
  EXPECT_EQ(NF(g, rpsum), NF(g, rptotal));
}

// Also show the boundary offset for the second tensor is aligned.
TEST(symbolic, common_concat_boundary_offset_is_aligned) {
  vkcnn::SymGraph g;
  auto W1 = g.var();
  int C = 16, B = 2, A = 16, K = C * B; // K multiple of A
  auto offset = g.alignUp(g.mul(W1, K), A);
  auto m = g.resolve(g.mod(offset, A));
  ASSERT_FALSE(m.isSymbolic());
  EXPECT_EQ(0, m.constant());
}

TEST(symbolic, common_vector_store_row_is_16_aligned_when_K_multiple_of_16) {
  vkcnn::SymGraph g;
  auto W = g.var();
  // Example: C=8, B=2 bytes => K=16
  int C = 8, B = 2, A = 16, K = C * B;
  auto m = g.resolve(g.mod(g.mul(W, K), A));
  ASSERT_FALSE(m.isSymbolic());
  EXPECT_EQ(0, m.constant());
}

TEST(symbolic, common_row_stride_offsets_stay_aligned) {
  vkcnn::SymGraph g;
  auto W = g.var(), Y = g.var();
  int C = 16, B = 2, A = 16,
      K = C * B; // K multiple of A ⇒ rowPitch is multiple of A
  auto rowPitch = g.alignUp(g.mul(W, K), A);
  auto off = g.mul(Y, rowPitch);
  auto m = g.resolve(g.mod(off, A));
  ASSERT_FALSE(m.isSymbolic());
  EXPECT_EQ(0, m.constant());
}

TEST(symbolic, common_tile_start_alignment) {
  vkcnn::SymGraph g;
  auto W = g.var(), Y = g.var(), T = g.var();
  // Choose A divisible by K; then taking x = T * (A/K) ensures x*K is multiple
  // of A.
  int C = 16, B = 1, A = 64, K = C * B;      // K=16, A/K=4
  auto x = g.mul(T, A / K);                  // x = T*(A/K)
  auto rowPitch = g.alignUp(g.mul(W, K), A); // == K*alignUp(W, A/K)
  auto off = g.add(g.mul(Y, rowPitch), g.mul(x, K));
  auto m = g.resolve(g.mod(off, A));
  ASSERT_FALSE(m.isSymbolic());
  EXPECT_EQ(0, m.constant());
}

TEST(symbolic, common_concat_many_when_K_multiple_of_A) {
  vkcnn::SymGraph g;
  auto W1 = g.var(), W2 = g.var(), W3 = g.var();
  int C = 16, B = 2, A = 16, K = C * B; // K multiple of A
  auto sumPads =
      g.add(g.add(g.alignUp(g.mul(W1, K), A), g.alignUp(g.mul(W2, K), A)),
            g.alignUp(g.mul(W3, K), A));
  auto allInOne = g.alignUp(g.mul(g.add(g.add(W1, W2), W3), K), A);
  EXPECT_EQ(NF(g, sumPads), NF(g, allInOne));
}

TEST(symbolic, common_concat_byte_offset_is_A_aligned) {
  vkcnn::SymGraph g;
  auto W1 = g.var(), H = g.var();
  int C = 8, B = 2, A = 16, K = C * B; // K=16 ⇒ rows already aligned to 16
  auto rowPitch = g.alignUp(g.mul(g.var() /*Wdummy*/, K),
                            A); // any width, but K multiple of A
  (void)rowPitch;               // not needed here, included to show pattern
  auto offsetBytes = g.alignUp(g.mul(W1, K), A);
  auto m = g.resolve(g.mod(offsetBytes, A));
  ASSERT_FALSE(m.isSymbolic());
  EXPECT_EQ(0, m.constant());
  (void)H; // H often participates in full image strides; omitted here
}

// (B*X) / (k*B) == floor(X/k)   — structurally: cancels B, leaves div by k
TEST(symbolic, common_peel_common_symbol_then_div_by_const_and_sym) {
  SymGraph g;
  auto B = g.var(), X = g.var();
  EXPECT_EQ(g.div(g.mul(B, X), g.mul(4, B)), g.div(X, 4));
}

// (B*X + B*Y) / (k*B) == floor((X+Y)/k)
TEST(symbolic, common_peel_common_symbol_of_affine_then_div_by_const_and_sym) {
  SymGraph g;
  auto B = g.var(), X = g.var(), Y = g.var();
  EXPECT_EQ(g.div(g.add(g.mul(B, X), g.mul(B, Y)), g.mul(3, B)),
            g.div(g.add(X, Y), 3));
}
// -----------------------------------------------------------------------------
// 6) Pooling K=1,P=0,D=1 on aligned width then upsample restores width (bytes).
//    This is your earlier idea but phrased in bytes with constants only.
// -----------------------------------------------------------------------------

TEST(symbolic, common_pool_k1_then_scale_back_preserves_row_bytes) {
  SymGraph g; auto W=g.var();
  int S=4, C=16, B=1, A=128, K=C*B;          // A divisible by K and by S*K
  auto alignedW   = g.alignUp(W, A / K);    // width aligned so that bytes are A-aligned
  auto pooledW    = g.pool(alignedW, 1, 0, S, 1); // == ceil(alignedW/S)
  auto upW        = g.mul(pooledW, S);
  auto rowBytes0  = g.mul(alignedW, K);
  auto rowBytes1  = g.mul(upW, K);
  EXPECT_EQ(NF(g,rowBytes0), NF(g,rowBytes1));     // exact width restored in bytes
  // Also show it stays A-aligned:
  auto m = g.resolve(g.mod(rowBytes1, A));
  ASSERT_FALSE(m.isSymbolic());
  EXPECT_EQ(0, m.constant());
}

// mod of any multiple: (B*E) % B == 0 ; sum of multiples too
TEST(symbolic, common_mod_of_multiples_vanishes) {
  SymGraph g;
  auto B = g.var(), E = g.var(), X = g.var(), Y = g.var();
  auto z1 = g.resolve(g.mod(g.mul(B, E), B));
  ASSERT_FALSE(z1.isSymbolic());
  EXPECT_EQ(0, z1.constant());

  auto z2 = g.resolve(g.mod(g.add(g.mul(B, X), g.mul(B, Y)), B));
  ASSERT_FALSE(z2.isSymbolic());
  EXPECT_EQ(0, z2.constant());
}

// combine: upsample(pool(aligned,S)) == aligned for K=1,P=0,D=1
TEST(symbolic, common_pool_k1_then_scale_back_on_aligned) {
  SymGraph g;
  auto S = g.var(), X = g.var();             // S>0 understood from / by design
  auto aligned = g.mul(S, X);                // already a multiple of S
  auto pooled = g.pool(aligned, 1, 0, S, 1); // == ceil(aligned/S) == X
  EXPECT_EQ(g.mul(pooled, S), aligned);
}


// -----------------------------------------------------------------------------
// 1) Row-pitch alignment equivalence (NHWC), with constants only
//    If A is divisible by K := C*B (bytes per pixel), then:
//    alignUp(W*K, A) == K * alignUp(W, A/K)
// -----------------------------------------------------------------------------
TEST(symbolic, common_row_pitch_align_equivalence_A_divisible_by_K) {
  SymGraph g; auto W = g.var();
  // Example: C=16 channels, B=1 byte/chan, A=64 bytes alignment.
  int C = 16, B = 1, A = 64, K = C*B; // K=16, A/K=4
  auto lhs = g.alignUp(g.mul(W, K), A);
  auto rhs = g.mul(g.alignUp(W, A / K), K);
  EXPECT_EQ(NF(g,lhs), NF(g,rhs));
}



// -----------------------------------------------------------------------------
// 7) AlignUp equivalence for bytes vs. pixels when A divisible by K:
//    alignUp(W*K, A)/K == alignUp(W, A/K)
// -----------------------------------------------------------------------------
TEST(symbolic, common_align_equivalence_bytes_vs_pixels) {
  SymGraph g; auto W=g.var();
  int C=16, B=1, A=64, K=C*B; // A/K=4
  auto lhs = g.div(g.alignUp(g.mul(W, K), A), K);
  auto rhs = g.alignUp(W, A / K);
  EXPECT_EQ(NF(g,lhs), NF(g,rhs));
}
