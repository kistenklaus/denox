#include "vkcnn/common/symbolic/SymGraph.hpp"
#include <gtest/gtest.h>

using namespace vkcnn;

#define NF(g, e) (g).resolve((e))
#define EQ_NF(g, a, b) EXPECT_EQ(NF(g, a), NF(g, b))

TEST(symbolic, modsolve_const_const_nested) {
  vkcnn::SymGraph g;
  // (13 % 7) % 5 = 6 % 5 = 1
  auto expr = g.mod(g.mod(13, 7), 5);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(1, r.constant());
}

TEST(symbolic, modsolve_const_div_then_mod) {
  vkcnn::SymGraph g;
  // (29 / 4) % 5 = 7 % 5 = 2
  auto expr = g.mod(g.div(29, 4), 5);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(2, r.constant());
}

// ---------- AFFINE % CONST (allDiv path) ----------

TEST(symbolic, modsolve_affine_allDiv_zero) {
  vkcnn::SymGraph g;
  auto X = g.var();
  // (8*X) % 8 == 0
  auto expr = g.mod(g.mul(8, X), 8);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(0, r.constant());
}

TEST(symbolic, modsolve_affine_allDiv_const_kept) {
  vkcnn::SymGraph g;
  auto X = g.var();
  // (8*X + 6) % 8 == 6
  auto expr = g.mod(g.add(g.mul(8, X), 6), 8);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(6, r.constant());
}

TEST(symbolic, modsolve_affine_coeffs_mixed_allDiv) {
  vkcnn::SymGraph g;
  auto X = g.var();
  // (16*X + 24) % 8 == 0
  auto expr = g.mod(g.add(g.mul(16, X), 24), 8);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(0, r.constant());
}

// ---------- ADD / SUB UNDER MOD M ----------

TEST(symbolic, modsolve_add_merges_and_reduces) {
  vkcnn::SymGraph g;
  auto X = g.var();
  // ((6*X) + (10*X)) % 8 == (16*X) % 8 == 0
  auto expr = g.mod(g.add(g.mul(6, X), g.mul(10, X)), 8);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(0, r.constant());
}

TEST(symbolic, modsolve_sub_by_multiple_of_m_is_noop_mod_m) {
  vkcnn::SymGraph g;
  auto X = g.var();
  // (X - 16) % 8  has same residue class as  X % 8, but not constant.
  // Instead build something that collapses:
  // (9*X - X) % 8 == (8*X) % 8 == 0
  auto expr = g.mod(g.sub(g.mul(9, X), X), 8);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(0, r.constant());
}

TEST(symbolic, modsolve_mod_sub_constant_wraps) {
  vkcnn::SymGraph g;
  // (5 - 13) % 7 == (-8) % 7 == 6
  auto expr = g.mod(g.sub(5, 13), 7);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(6, r.constant());
}

// ---------- MUL UNDER MOD M (const * symbolic) ----------

TEST(symbolic, modsolve_mod_mul_const_symbolic_reduces) {
  vkcnn::SymGraph g;
  auto X = g.var();
  // (12*X + 3) % 6 == 3
  auto expr = g.mod(g.add(g.mul(12, X), 3), 6);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(3, r.constant());
}

TEST(symbolic, modsolve_mod_mul_then_mod_zero) {
  vkcnn::SymGraph g;
  auto X = g.var();
  // ((4*X) * 2) % 8 == (8*X) % 8 == 0
  auto expr = g.mod(g.mul(g.mul(4, X), 2), 8);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(0, r.constant());
}

// ---------- DIV EDGE CASES ----------

TEST(symbolic, modsolve_mod_div_zero_over_symbolic_then_mod) {
  vkcnn::SymGraph g;
  auto A = g.var();
  // (0 / A) % 5 == 0
  auto expr = g.mod(g.div(0, A), 5);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(0, r.constant());
}

TEST(symbolic, modsolve_mod_affine_div_then_mod_allDiv) {
  vkcnn::SymGraph g;
  auto X = g.var();
  // ((4*X + 2) / 2) % 2 == (2*X + 1) % 2 == 1  (since 2X ≡ 0 mod 2)
  auto expr = g.mod(g.div(g.add(g.mul(4, X), 2), 2), 2);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(1, r.constant());
}

// ---------- NESTED MOD (constant collapsing path) ----------

TEST(symbolic, modsolve_mod_nested_mod_inner_constantizes) {
  vkcnn::SymGraph g;
  auto X = g.var();
  // ((4*X + 1) % 4) % 2  => 1 % 2 = 1
  auto inner = g.mod(g.add(g.mul(4, X), 1), 4);
  auto expr = g.mod(inner, 2);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(1, r.constant());
}

// ---------- “SHAPE” STYLE IDENTITIES ----------

TEST(symbolic, modsolve_mod_stride_multiple_is_zero_mod_stride) {
  vkcnn::SymGraph g;
  auto W = g.var();
  std::size_t stride = 4;
  // ((floor((W + s - 1)/s)) * s) % s == 0
  auto q = g.div(g.add(W, stride - 1), stride);
  auto n = g.mul(q, stride);
  auto expr = g.mod(n, stride);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(0, r.constant());
}

TEST(symbolic, modsolve_mod_pad_plus_stride_residue) {
  vkcnn::SymGraph g;
  auto W = g.var();
  std::size_t s = 6;
  std::size_t pad = 11;
  // (s*W + pad) % s == pad % s == 5
  auto expr = g.mod(g.add(g.mul(s, W), pad), s);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(pad % s, r.constant());
}

// =====================
// CONSTANT / BASIC FLOW
// =====================

TEST(symbolic, modsolve_mod_const_const_nested_big) {
  vkcnn::SymGraph g;
  // ((137 % 29) % 23) = (21 % 23) = 21
  auto expr = g.mod(g.mod(137, 29), 23);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(21, r.constant());
}

TEST(symbolic, modsolve_mod_const_div_then_mod_large) {
  vkcnn::SymGraph g;
  // (1001 / 7) % 13 = 143 % 13 = 0
  auto expr = g.mod(g.div(1001, 7), 13);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(0, r.constant());
}

// =====================
// ALL-DIVISIBLE AFFINE
// =====================

TEST(symbolic, modsolve_affine_allDiv_mixed_large) {
  vkcnn::SymGraph g;
  auto X = g.var();
  // (24*X + 40) % 8 == 0
  auto expr = g.mod(g.add(g.mul(24, X), 40), 8);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(0, r.constant());
}

TEST(symbolic, modsolve_affine_allDiv_constant_survives) {
  vkcnn::SymGraph g;
  auto X = g.var();
  // (12*X + 29) % 12 == 5
  auto expr = g.mod(g.add(g.mul(12, X), 29), 12);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(29 % 12, r.constant());
}

// =====================
// ADD/SUB UNDER MOD M
// =====================

TEST(symbolic, modsolve_add_big_coeff_cancellation) {
  vkcnn::SymGraph g;
  auto A = g.var();
  // (14*A + 18*A + 7) % 16 = (32*A + 7) % 16 = 7
  auto expr = g.mod(g.add(g.mul(14, A), g.add(g.mul(18, A), 7)), 16);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(7, r.constant());
}

TEST(symbolic, modsolve_sub_constant_negative_wrap) {
  vkcnn::SymGraph g;
  // (5 - 29) % 12 = (-24) % 12 = 0
  auto expr = g.mod(g.sub(5, 29), 12);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(0, r.constant());
}

// =====================
// MUL UNDER MOD (const×sym)
// =====================

TEST(symbolic, modsolve_mul_const_symbolic_chain) {
  vkcnn::SymGraph g;
  auto X = g.var();
  // ((9*X)*14 + 35) % 7 == (126*X + 35) % 7 == 0
  auto expr = g.mod(g.add(g.mul(g.mul(9, X), 14), 35), 7);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(0, r.constant());
}

TEST(symbolic, modsolve_mul_const_symbolic_nonzero) {
  vkcnn::SymGraph g;
  auto X = g.var();
  // (10*X + 17) % 9 == 17 % 9 == 8
  auto expr = g.mod(g.add(g.mul(9, X), 17), 9);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(8, r.constant());
}

// =====================
// DIV → MOD (affine_div success path)
// =====================

TEST(symbolic, modsolve_div_affine_success_then_mod_reduce) {
  vkcnn::SymGraph g;
  auto X = g.var();
  // ((6*X + 4)/2) % 5  == (3*X + 2) % 5
  // Under mod 5, 3*X isn't constant; but (3*X + 2) % 5 has constant 2 when X ≡
  // 0 mod 5. We just check structure by matching against the explicit form:
  auto lhs = g.mod(g.div(g.add(g.mul(6, X), 4), 2), 5);
  auto rhs = g.mod(g.add(g.mul(3, X), 2), 5);
  EXPECT_EQ(rhs, lhs);
}

TEST(symbolic, modsolve_div_affine_then_mod_constantize) {
  vkcnn::SymGraph g;
  auto X = g.var();
  // ((4*X + 2)/2) % 2 == (2*X + 1) % 2 == 1
  auto expr = g.mod(g.div(g.add(g.mul(4, X), 2), 2), 2);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(1, r.constant());
}

// =====================
// NESTED MOD that collapses RHS via solver
// =====================

TEST(symbolic, modsolve_nested_mod_rhs_collapses_via_solver) {
  vkcnn::SymGraph g;
  auto X = g.var(), Y = g.var();
  // inner: (4*Y + 1) % 4 = 1
  // X % ( (4Y+1)%4 )  => X % 1  => 0
  auto inner = g.mod(g.add(g.mul(4, Y), 1), 4);
  auto expr = g.mod(X, inner); // lhs symbolic, rhs symbolic-but-collapses
  auto r =
      g.resolve(g.mod(expr, 9)); // outer mod m (your code rewraps to mod m)
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(0, r.constant());
}

// =====================
// GCD-BASED AFFINE % AFFINE COLLAPSE
// =====================

TEST(symbolic, modsolve_affine_mod_affine_scaled_result) {
  vkcnn::SymGraph g;
  auto A = g.var(), B = g.var();
  // Let P = (A + 2B + 3).
  // LHS = 5*P = 5A + 10B + 15
  // RHS = 3*P = 3A + 6B + 9
  // gcd(LHS)=5, gcd(RHS)=3, reduced forms equal → LHS % RHS = (5 % 3)*P = 2*P
  // Compare under a big modulus to avoid accidental coefficient wraps.
  auto LHS = g.add(g.add(g.mul(5, A), g.mul(10, B)), 15);
  auto RHS = g.add(g.add(g.mul(3, A), g.mul(6, B)), 9);
  auto left = g.mod(g.mod(LHS, RHS), 97);
  auto right = g.mod(g.add(g.add(g.mul(2, A), g.mul(4, B)), 6), 97);
  EXPECT_EQ(right, left);
}

TEST(symbolic, modsolve_affine_mod_affine_zero_result) {
  vkcnn::SymGraph g;
  auto A = g.var(), B = g.var();
  // LHS = 6*(A + 2B + 3) ; RHS = 3*(A + 2B + 3) → LHS % RHS = 0
  auto LHS = g.add(g.add(g.mul(6, A), g.mul(12, B)), 18);
  auto RHS = g.add(g.add(g.mul(3, A), g.mul(6, B)), 9);
  auto expr = g.mod(g.mod(LHS, RHS), 101);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(0, r.constant());
}

// =====================
// MOD-MOD absorption when RHS ≡ 0 (mod m)
// =====================

TEST(symbolic, modsolve_mod_rhs_multiple_of_m_is_noop) {
  vkcnn::SymGraph g;
  auto A = g.var();
  // (A % (16)) % 8 == A % 8   (uses rhs.constant() % m == 0 fast path)
  auto left = g.mod(g.mod(A, 16), 8);
  auto right = g.mod(A, 8);
  EXPECT_EQ(right, left);
}

// =====================
// “SHAPE ALGEBRA” DEEPER MIX
// =====================

TEST(symbolic, modsolve_conv_stride_block_alignment) {
  vkcnn::SymGraph g;
  auto W = g.var();
  // q = floor((W + 7)/8); n = q*8;  n % 8 == 0
  auto q = g.div(g.add(W, 7), 8);
  auto n = g.mul(q, 8);
  auto expr = g.mod(n, 8);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(0, r.constant());
}

TEST(symbolic, modsolve_conv_stride_pad_mix) {
  vkcnn::SymGraph g;
  auto W = g.var();
  std::size_t s = 6, pad = 23; // pad % s = 5
  // (s*W + pad) % s == pad % s
  auto expr = g.mod(g.add(g.mul(s, W), pad), s);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(pad % s, r.constant());
}

// =====================
// BAIL-INTENT TESTS (should remain symbolic)
// =====================

TEST(symbolic, modsolve_mul_symbolic_symbolic_bails) {
  vkcnn::SymGraph g;
  auto X = g.var(), Y = g.var();
  auto expr = g.mod(g.mul(X, Y), 2); // your modsolve_mul bails on sym×sym
  auto r = g.resolve(expr);
  EXPECT_TRUE(r.isSymbolic()); // stays symbolic
}

TEST(symbolic, modsolve_div_symbolic_symbolic_bails) {
  vkcnn::SymGraph g;
  auto X = g.var(), Y = g.var();
  auto expr = g.mod(g.div(g.add(X, 1), g.add(Y, 2)), 7);
  auto r = g.resolve(expr);
  EXPECT_TRUE(r.isSymbolic()); // stays symbolic
}

// ============ BASIC PARITY CASE ============

TEST(symbolic, modsolve_parity_evenizer_k2) {
  vkcnn::SymGraph g;
  auto X = g.var();
  // (X + (2 - (X % 2))) % 2 == 0
  auto expr = g.mod(g.add(X, g.sub(2, g.mod(X, 2))), 2);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(0, r.constant());
}

// ============ GENERAL k (same outer modulus) ============

TEST(symbolic, modsolve_evenizer_general_k_equal_modulus) {
  vkcnn::SymGraph g;
  auto E = g.var();
  std::size_t k = 12;
  // (E + (k - (E % k))) % k == 0
  auto expr = g.mod(g.add(E, g.sub((int)k, g.mod(E, (int)k))), (int)k);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(0, r.constant());
}

// ============ m | k (outer modulus divides inner k) ============

TEST(symbolic, modsolve_evenizer_outer_divides_inner) {
  vkcnn::SymGraph g;
  auto E = g.var();
  std::size_t k = 12, m = 4; // m | k
  // Since k ≡ 0 (mod m), we still get 0:
  // (E + (k - (E % k))) % m == 0
  auto expr = g.mod(g.add(E, g.sub((int)k, g.mod(E, (int)k))), (int)m);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(0, r.constant());
}

// ============ Affine E also works ============

TEST(symbolic, modsolve_evenizer_affine_E_k2) {
  vkcnn::SymGraph g;
  auto X = g.var();
  auto E = g.add(g.mul(5, X), 3); // E = 5X + 3
  auto expr = g.mod(g.add(E, g.sub(2, g.mod(E, 2))), 2);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(0, r.constant());
}

TEST(symbolic, modsolve_evenizer_affine_E_general_k) {
  vkcnn::SymGraph g;
  auto X = g.var();
  auto E = g.add(g.mul(5, X), 3);
  int k = 8;
  auto expr = g.mod(g.add(E, g.sub(k, g.mod(E, k))), k);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(0, r.constant());
}

// ============ Add extra multiples of k (should still be 0) ============

TEST(symbolic, modsolve_evenizer_plus_multiple_of_k) {
  vkcnn::SymGraph g;
  auto X = g.var();
  int k = 10;
  // + 3k doesn't change residue mod k
  auto expr = g.mod(g.add(g.add(X, g.sub(k, g.mod(X, k))), 3 * k), k);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(0, r.constant());
}

// ============ Negative case: m does NOT divide k (not guaranteed 0)
// ============

TEST(symbolic, modsolve_evenizer_m_not_dividing_k_stays_symbolic) {
  vkcnn::SymGraph g;
  auto E = g.var();
  int k = 6, m = 4; // m !| k
  // (E + (6 - (E % 6))) % 4 is NOT necessarily 0; should remain symbolic.
  auto expr = g.mod(g.add(E, g.sub(k, g.mod(E, k))), m);
  auto r = g.resolve(expr);
  EXPECT_TRUE(r.isSymbolic()); // we don't claim a constant here
}


TEST(symbolic, modsolve_mod16_mod4) {
  vkcnn::SymGraph g;
  auto X = g.var();
  auto a = g.mod(g.mod(X, 16), 4);
  auto b = g.mod(X, 4);
  EXPECT_EQ(a, b);
}

TEST(symbolic, modsolve_sub_mod16_mod4) {
  vkcnn::SymGraph g;
  auto X = g.var();
  auto a = g.mod(g.mod(X, 16), 4);
  auto b = g.mod(X, 4);
  EXPECT_EQ(a, b);
}

TEST(symbolic, modsolve_sub_mod16_mod4_add_const) {
  vkcnn::SymGraph g;
  auto X = g.var();
  auto U = g.add(g.sub(X, g.mod(X, 16)), 14);
  auto m = g.resolve(g.mod(U, 4));
  ASSERT_FALSE(m.isSymbolic());
  EXPECT_EQ(2, m.constant());
}



TEST(symbolic, modsolve_div_lifting_Y_mod_2) {
  vkcnn::SymGraph g;
  auto W = g.var();

  // U = W - (W mod 16) + 14
  auto U = g.add(g.sub(W, g.mod(W, 16)), 14);

  // Sanity: U mod 4 == 2
  auto U_mod4 = g.resolve(g.mod(U, 4));
  ASSERT_FALSE(U_mod4.isSymbolic());
  EXPECT_EQ(2, U_mod4.constant());

  // Y = U div 2  ⇒  Y mod 2 must be 1 (requires lifting to modulus 4
  // internally)
  auto Y = g.div(U, 2);
  auto Y_mod2 = g.resolve(g.mod(Y, 2));
  ASSERT_FALSE(Y_mod2.isSymbolic());
  EXPECT_EQ(1, Y_mod2.constant());

  // And (Y - 1) is even
  auto even_mod = g.resolve(g.mod(g.sub(Y, 1), 2));
  ASSERT_FALSE(even_mod.isSymbolic());
  EXPECT_EQ(0, even_mod.constant());
}

TEST(symbolic, modsolve_canonicalize_mod_m_n) {
  vkcnn::SymGraph g;
  auto X = g.var();
  // a = (3(X mod 16) + 5) mod 4
  auto a = g.mod(g.add(g.mul(3, g.mod(X, 16)), 5), 4);
  // b = (3X + 5) mod 4
  auto b = g.mod(g.add(g.mul(3, X), 5), 4);
  // Because 16 % 4 == 0 => a == b.
  EXPECT_EQ(a, b);
}

TEST(symbolic, modsolve_div_lifting_div4_mod4) {
  vkcnn::SymGraph g;
  auto W = g.var();

  auto U = g.add(g.sub(W, g.mod(W, 16)), 14); // U ≡ 14 (mod 16)
  auto Z = g.div(U, 4);                       // floor((16k+14)/4) = 4k+3

  // Z mod 4 must be 3 (requires lifting to modulus 16 internally)
  auto Z_mod4 = g.resolve(g.mod(Z, 4));
  ASSERT_FALSE(Z_mod4.isSymbolic());
  EXPECT_EQ(3, Z_mod4.constant());
}


// ============================================================================
// Constant-modulo collapse tests
// Suite: symbolic
// Prefix: modsolve_
// ============================================================================

TEST(symbolic, modsolve_multiples_drop_to_zero_simple) {
  SymGraph g; auto X=g.var();
  auto r = g.resolve(g.mod(g.mul(16, X), 16));
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(0, r.constant());
}

TEST(symbolic, modsolve_sum_of_multiples_plus_c_collapses) {
  SymGraph g; auto X=g.var(), Y=g.var(), Z=g.var();
  // (6X + 12Y + 18Z + 23) % 6 == 23 % 6 == 5
  auto expr = g.mod(g.add(g.add(g.add(g.mul(6,X), g.mul(12,Y)), g.mul(18,Z)), 23), 6);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(5, r.constant());
}

TEST(symbolic, modsolve_mul_of_div_term_vanishes) {
  SymGraph g; auto X=g.var();
  // 14 * floor(X/7) is a multiple of 7 ⇒ remainder 0
  auto expr = g.mod(g.mul(14, g.div(X, 7)), 7);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(0, r.constant());
}

TEST(symbolic, modsolve_div_scaled_plus_const_collapses) {
  SymGraph g; auto X=g.var();
  // (14 * floor((X+6)/7) + 41) % 7 == 41 % 7 == 6
  auto expr = g.mod(g.add(g.mul(14, g.div(g.add(X, 6), 7)), 41), 7);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(6, r.constant());
}

TEST(symbolic, modsolve_alignUp_times_k_plus_c_collapses) {
  SymGraph g; auto X=g.var();
  int A = 64, k = 3, c = 10;
  // alignUp(X,A) = A * ceil(X/A) ⇒ multiple of A
  auto expr = g.mod(g.add(g.mul(k, g.alignUp(X, A)), c), A);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(c % A, r.constant());
}

TEST(symbolic, modsolve_alignUp_self_is_multiple) {
  SymGraph g; auto X=g.var();
  int A = 32;
  auto expr = g.mod(g.alignUp(X, A), A);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(0, r.constant());
}

TEST(symbolic, modsolve_nested_mod_coeffs_kill_terms_mod2) {
  SymGraph g; auto X=g.var(), Y=g.var();
  // ((X%8)*4 + (Y%4)*8 + 29) % 4 == 29 % 4 == 1
  auto expr = g.mod(g.add(g.add(g.mul(4, g.mod(X, 8)), g.mul(8, g.mod(Y, 4))), 29), 4);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(1, r.constant());
}

TEST(symbolic, modsolve_nested_mod_coeffs_kill_terms_mod3) {
  SymGraph g; auto X=g.var();
  // ((X%12)*15 + 17) % 3 == 17 % 3 == 2
  auto expr = g.mod(g.add(g.mul(15, g.mod(X, 12)), 17), 3);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(2, r.constant());
}

TEST(symbolic, modsolve_large_mix_all_symbol_terms_vanish) {
  SymGraph g; auto X=g.var(), Y=g.var(), Z=g.var();
  // (64X + 48(Y+Z) + 510) % 16 == 510 % 16 == 14
  auto expr = g.mod(
      g.add(g.add(g.mul(64, X), g.mul(48, g.add(Y, Z))), 510),
      16);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(14, r.constant());
}

TEST(symbolic, modsolve_power_of_two_style) {
  SymGraph g; auto X=g.var();
  // (64*X + 22) % 64 == 22
  auto expr = g.mod(g.add(g.mul(64, X), 22), 64);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(22, r.constant());
}

TEST(symbolic, modsolve_negative_constant_wrapped) {
  SymGraph g;
  // (-3) % 8 == 5  (with Euclidean remainder)
  auto expr = g.mod(g.add(0, -3), 8);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(5, r.constant());
}

TEST(symbolic, modsolve_combo_multiples_and_constants) {
  SymGraph g; auto X=g.var(), Y=g.var();
  // (20X + 30Y + 77) % 10 == 77 % 10 == 7
  auto expr = g.mod(g.add(g.add(g.mul(20, X), g.mul(30, Y)), 77), 10);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(7, r.constant());
}

TEST(symbolic, modsolve_pool_then_scale_modS_zero) {
  SymGraph g; auto S=g.var(), X=g.var();
  // pool(S*X,1,0,S,1) == ceil((S*X)/S) == X, then S*X % S == 0
  auto expr = g.mod(g.mul(g.pool(g.mul(S, X), 1, 0, S, 1), S), S);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(0, r.constant());
}

TEST(symbolic, modsolve_mixed_mods_coeffs_kill_everything_mod2) {
  SymGraph g; auto X=g.var();
  // (8X + 4*(X%2) + 30) % 2 == 0
  auto expr = g.mod(g.add(g.add(g.mul(8, X), g.mul(4, g.mod(X, 2))), 30), 2);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(0, r.constant());
}

TEST(symbolic, modsolve_many_multiples_then_small_const) {
  SymGraph g; auto A=g.var(), B=g.var(), C=g.var();
  // (96A + 40B + 72C + 11) % 8 == 11 % 8 == 3
  auto expr = g.mod(g.add(g.add(g.add(g.mul(96, A), g.mul(40, B)), g.mul(72, C)), 11), 8);
  auto r = g.resolve(expr);
  ASSERT_FALSE(r.isSymbolic());
  EXPECT_EQ(3, r.constant());
}
