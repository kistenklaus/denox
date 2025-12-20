#include "symbolic/SymGraph.hpp"
#include <gtest/gtest.h>

// ---------- Helpers: normalize-then-compare & idempotence ----------
#define NF(g, e) (g).resolve((e))
#define EQ_NF(g, a, b) EXPECT_EQ(NF(g, a), NF(g, b))

using namespace denox::compiler;

TEST(symbolic, unsolvable_div_not_divisible_bails) {
  SymGraph g;
  auto A = g.var();
  auto e = g.add(g.mul(6, A), 8);  // 6A+8
  auto q = g.resolve(g.div(e, 4)); // 6A/4 not integer → symbolic
  EXPECT_TRUE(q.isSymbolic());
}

TEST(symbolic, unsolvable_div_by_var_nontrivial) {
  SymGraph g;
  auto A = g.var();
  // (2A + 1) / A  -> not a constant (your A/A special-case doesn't apply)
  auto q = g.resolve(g.div(g.add(g.mul(2, A), 1), A));
  EXPECT_TRUE(q.isSymbolic());
}

TEST(symbolic, unsolvable_div_const_not_all_divisible) {
  SymGraph g;
  auto A = g.var();
  // (A + 1) / 2  -> some terms not divisible by 2 => bail
  auto q = g.resolve(g.div(g.add(A, 1), 2));
  EXPECT_TRUE(q.isSymbolic());
}

TEST(symbolic, unsolvable_mod_const_not_all_divisible) {
  SymGraph g;
  auto A = g.var();
  // (A + 1) % 2  -> coef 1 not divisible by 2 => bail
  auto r = g.resolve(g.mod(g.add(A, 1), 2));
  EXPECT_TRUE(r.isSymbolic());
}

TEST(symbolic, unsolvable_const_mod_affine_bails) {
  SymGraph g;
  auto A = g.var();
  // 7 % (A + 2)  -> rhs non-constant => bail in Z-affine
  auto r = g.resolve(g.mod(7, g.add(A, 2)));
  EXPECT_TRUE(r.isSymbolic());
}

TEST(symbolic, unsolvable_affine_mod_affine_different_base_bails) {
  SymGraph g;
  auto A = g.var();
  // (A + 1) % (A + 3)  -> bases differ, gcd-rule doesn't apply => bail
  auto r = g.resolve(g.mod(g.add(A, 1), g.add(A, 3)));
  EXPECT_TRUE(r.isSymbolic());
}

TEST(symbolic, unsolvable_div_const_not_divisible_some_terms) {
  SymGraph g;
  auto A = g.var();
  // (4A + 2) / 3  -> coefficients not all divisible by 3 => bail
  auto q = g.resolve(g.div(g.add(g.mul(4, A), 2), 3));
  EXPECT_TRUE(q.isSymbolic());
}

TEST(symbolic, unsolvable_compound_bail_div_then_mod) {
  SymGraph g;
  auto A = g.var();
  // (A + 2)/2  already bails -> symbol; then %2 should also bail in Z-affine
  auto r = g.resolve(g.mod(g.div(g.add(A, 2), 2), 2));
  EXPECT_TRUE(r.isSymbolic());
}

TEST(symbolic, unsolvable_mod_same_base_nonzero_remainder_is_affine_not_const) {
  SymGraph g;
  auto A = g.var();
  auto E = g.add(A, 1); // affine base
  // (3E) % (2E) = (3%2) * E = E  -> symbolic (non-constant) but simplified
  auto r = g.resolve(g.mod(g.mul(3, E), g.mul(2, E)));
  EXPECT_EQ(r, E);
  EXPECT_TRUE(r.isSymbolic());
}

TEST(symbolic, unsolvable_parity) {
  SymGraph g;
  auto A = g.var();
  auto r = g.resolve(g.mod(g.add(A, 1), 2)); // depends on A parity
  EXPECT_TRUE(r.isSymbolic());
}

TEST(symbolic, unsolvable_no_cancel_when_not_multiple) {
  SymGraph g;
  auto A = g.var(), B = g.var();
  auto lhs = g.div(g.add(B, g.sub(A, 1)), A); // (B + A - 1)/A
  EXPECT_NE(B, lhs);                          // not equal to B
}

TEST(symbolic, unsolvable_no_factorization_AB_plus_A_over_AC) {
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var();
  auto lhs = g.div(g.add(g.mul(A, B), A), g.mul(A, C)); // (AB + A)/(AC)
  auto wrong = g.div(g.add(1, B), C); // (1+B)/C  (needs factoring A)
  EXPECT_NE(wrong, lhs);              // we don't factor
}

TEST(symbolic, unsolvable_not_zero_A_over_AB) {
  SymGraph g;
  auto A = g.var(), B = g.var();
  auto expr = g.div(A, g.mul(A, B)); // A/(A*B) == 1/B, not 0
  EXPECT_NE(g.mul(0, A), expr);
  EXPECT_EQ(g.div(1, B), expr);
}

TEST(symbolic, unsolvable_not_B_AB_over_AC) {
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var();
  auto lhs = g.div(g.mul(A, B), g.mul(A, C)); // AB/AC
  EXPECT_NE(B, lhs);                          // only B/C is correct
}

TEST(symbolic, unsolvable_div_const_over_symbol_not_zero) {
  SymGraph g;
  auto X = g.var();
  auto expr = g.div(1, X);
  EXPECT_NE(g.mul(0, X), expr); // 1/X is not 0 in general
}

TEST(symbolic, unsolvable_no_discharge_wrong_residual) {
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var();
  auto lhs = g.div(g.sub(C, 1),
                   g.mul(A, B)); // (C - 1)/(AB)  (residual is C-1, not AB-1)
  EXPECT_NE(g.mul(0, A), lhs);   // must not drop to 0
}

TEST(symbolic, unsolvable_disjoint_no_cancel) {
  SymGraph g;
  auto A = g.var(), B = g.var(), D = g.var();
  auto lhs = g.div(g.mul(A, B), D); // (A*B)/D, no common factor
  auto wrong = g.div(B, D);         // pretending A cancels
  EXPECT_NE(wrong, lhs);
}

TEST(symbolic, unsolvable_nested_div_not_flattened_to_A_over_B) {
  SymGraph g;
  auto A = g.var(), B = g.var();
  auto lhs = g.div(g.div(A, B), B); // (A/B)/B
  EXPECT_NE(g.div(A, B), lhs);      // must be A/(B*B)
  EXPECT_EQ(g.div(A, g.mul(B, B)), lhs);
}

TEST(symbolic, unsolvable_prod_gcd_no_underflow_on_multiplicity) {
  SymGraph g;
  auto A = g.var(), C = g.var();
  auto lhs = g.div(g.mul(A, C), g.mul(C, C)); // (A*C)/(C*C) == A/C
  EXPECT_EQ(g.div(A, C), lhs);
  EXPECT_NE(A, lhs); // not A (since C remains)
}

TEST(symbolic, unsolvable_modd_depends_on_symbol) {
  SymGraph g;
  auto A = g.var(), B = g.var();
  // (4A + 3B + 1) % 2  depends on B (3%2=1) -> cannot fold to affine
  auto e = g.add(g.add(g.mul(4, A), g.mul(3, B)), 1);
  auto m = g.mod(e, 2);
  // Expect: still symbolic (i.e., affine_mod returned nullopt and you created a
  // Mod node)
  auto r = g.resolve(m);
  EXPECT_TRUE(r.isSymbolic()); // or however you check non-constant here
}

TEST(symbolic, unsolvable_mod_affine_sum_not_factored) {
  SymGraph g;
  auto A = g.var(), B = g.var();
  auto expr =
      g.mod(g.add(g.mul(A, B), A), A); // (A*B + A) % A == 0 mathematically,
                                       // but we are not factoring sums here.
  auto zero = g.mul(0, A);
  EXPECT_NE(zero, expr);
}

TEST(symbolic, unsolvable_mod_disjoint_products) {
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var();
  auto expr = g.mod(g.mul(A, B), C); // (A*B) % C  (no common factor guarantee)
  // stay as non-affine mod
  EXPECT_TRUE(g.resolve(expr).isSymbolic());
}

TEST(symbolic, unsolvable_mod_nested_symbolic_mixed) {
  SymGraph g;
  auto X = g.var(), A = g.var(), B = g.var();
  auto expr = g.mod(g.mod(X, A), B); // cannot combine without knowing gcd(A,B)
  EXPECT_TRUE(g.resolve(expr).isSymbolic());
}

TEST(symbolic, unsolvable_mod_nested_symbolic_must_not_drop_outer) {
  SymGraph g;
  auto X = g.var(), A = g.var(), B = g.var();
  auto inner = g.mod(X, A);
  auto expr = g.mod(inner, B); // (X % A) % B
  EXPECT_NE(inner, expr);      // must NOT simplify to X % A (unless B==A)
  auto same = g.mod(inner, A);
  EXPECT_EQ(inner, same); // idempotence only when A==A
}

// --- Division/multiplication cancellation pitfalls ---

TEST(symbolic, unsolvable_div_mul_no_cancel_with_constant) {
  SymGraph g;
  auto X = g.var();
  // floor(X/2)*2 == X only if X is even. Do NOT simplify.
  EXPECT_NE(g.mul(g.div(X, 2), 2), X);
}

// --------------------------- Division / Ceil division
// ---------------------------

TEST(symbolic, unsolvable_iterated_cdiv_not_collapsible) {
  SymGraph g;
  auto X = g.var();
  auto A = g.var();
  auto B = g.var();
  // A,B!=0 (from cdiv). ceil(ceil(X/A)/B) != ceil(X/(A*B)) in general +
  // potential overflow in A*B.
  EXPECT_NE(g.cdiv(g.cdiv(X, A), B), g.cdiv(X, g.mul(A, B)));
}

// ------------------------------- Modulo traps
// ----------------------------------

TEST(symbolic, unsolvable_mod_add_without_final_mod) {
  SymGraph g;
  auto X = g.var();
  auto Y = g.var();
  auto A = g.var();
  // A!=0 (from %A). (X+Y)%A != (X%A + Y%A) in general, even though all
  // mods/divs imply A>0.
  EXPECT_NE(g.mod(g.add(X, Y), A), g.add(g.mod(X, A), g.mod(Y, A)));
}

TEST(symbolic, unsolvable_mod_mul_without_final_mod) {
  SymGraph g;
  auto X = g.var();
  auto Y = g.var();
  auto A = g.var();
  // A!=0 (from %A). (X*Y)%A != (X%A)*(Y%A) in general; overflow risks too.
  EXPECT_NE(g.mod(g.mul(X, Y), A), g.mul(g.mod(X, A), g.mod(Y, A)));
}

TEST(symbolic, unsolvable_nested_mod_different_moduli) {
  SymGraph g;
  auto X = g.var();
  auto A = g.var();
  auto B = g.var();
  // A,B>0 from mods. (X%A)%B != X%B in general unless strong relations between
  // A and B.
  EXPECT_NE(g.mod(g.mod(X, A), B), g.mod(X, B));
}

// ------------------------------- Floor/Ceil laws
// -------------------------------

TEST(symbolic, unsolvable_floor_div_not_distributive_over_add) {
  SymGraph g;
  auto X = g.var();
  auto Y = g.var();
  auto A = g.var();
  // A>0 (from divisions). floor((X+Y)/A) != floor(X/A)+floor(Y/A) in general.
  EXPECT_NE(g.div(g.add(X, Y), A), g.add(g.div(X, A), g.div(Y, A)));
}

TEST(symbolic, unsolvable_ceil_div_not_distributive_over_add) {
  SymGraph g;
  auto X = g.var();
  auto Y = g.var();
  auto A = g.var();
  // A>0 from cdiv. ceil((X+Y)/A) != ceil(X/A)+ceil(Y/A) in general.
  EXPECT_NE(g.cdiv(g.add(X, Y), A), g.add(g.cdiv(X, A), g.cdiv(Y, A)));
}

// ------------------------------- Pooling formulas
// ------------------------------

TEST(symbolic, unsolvable_pool_vs_cpool_general) {
  SymGraph g;
  auto E = g.var();
  auto K = g.var();
  auto P = g.var();
  auto S = g.var();
  auto D = g.var();
  // S>0 implied by internal divs, but floor vs ceil outputs differ generally.
  EXPECT_NE(g.pool(E, K, P, S, D), g.cpool(E, K, P, S, D));
}

// ------------------------------- “Still symbolic”
// ------------------------------

TEST(symbolic, unsolvable_reciprocal_of_sum_remains_symbolic) {
  SymGraph g;
  auto X = g.var();
  auto Y = g.var();
  // 1/(X+Y) implies X+Y>0 and no overflow in (X+Y), but value not constant.
  auto expr = g.resolve(g.div(1, g.add(X, Y)));
  EXPECT_TRUE(expr.isSymbolic());
}

TEST(symbolic, unsolvable_reciprocal_of_mod_sum_remains_symbolic) {
  SymGraph g;
  auto X = g.var();
  auto A = g.var();
  auto B = g.var();
  // 1/( (X%A) + (B% A) ) enforces A>0 and sum>0 (no overflow), yet not a
  // constant.
  auto expr = g.resolve(g.div(1, g.add(g.mod(X, A), g.mod(B, A))));
  EXPECT_TRUE(expr.isSymbolic());
}

TEST(symbolic, unsolvable_nested_cdiv_composition_symbolic) {
  SymGraph g;
  auto X = g.var();
  auto A = g.var();
  auto B = g.var();
  // Both denominators >0; without more structure this shouldn't fold to a
  // constant or X.
  auto expr = g.resolve(g.cdiv(g.add(g.cdiv(X, A), g.cdiv(X, B)), g.add(A, B)));
  EXPECT_TRUE(expr.isSymbolic());
}

// ---- Division / Ceil division (A>0 is known, no overflow anywhere) ----

TEST(symbolic, unsolvable_div_sum_not_linear) {
  SymGraph g;
  auto X = g.var();
  auto Y = g.var();
  auto A = g.var();
  EXPECT_NE(g.div(g.add(X, Y), A), g.add(g.div(X, A), g.div(Y, A)));
}

TEST(symbolic, unsolvable_cdiv_sum_not_linear) {
  SymGraph g;
  auto X = g.var();
  auto Y = g.var();
  auto A = g.var();
  EXPECT_NE(g.cdiv(g.add(X, Y), A), g.add(g.cdiv(X, A), g.cdiv(Y, A)));
}

TEST(symbolic, unsolvable_cdiv_nested_not_chain) {
  SymGraph g;
  auto X = g.var();
  auto A = g.var();
  auto B = g.var();
  EXPECT_NE(g.cdiv(g.cdiv(X, A), B), g.cdiv(X, g.mul(A, B)));
}

TEST(symbolic, unsolvable_div_vs_cdiv) {
  SymGraph g;
  auto X = g.var();
  auto A = g.var();
  EXPECT_NE(g.div(X, A), g.cdiv(X, A));
}

// ---- Modulo (A>0 known from %A; no overflow) ----

TEST(symbolic, unsolvable_mod_add_needs_final_mod) {
  SymGraph g;
  auto X = g.var();
  auto Y = g.var();
  auto A = g.var();
  EXPECT_NE(g.mod(g.add(X, Y), A), g.add(g.mod(X, A), g.mod(Y, A)));
}

TEST(symbolic, unsolvable_mod_mul_needs_final_mod) {
  SymGraph g;
  auto X = g.var();
  auto Y = g.var();
  auto A = g.var();
  EXPECT_NE(g.mod(g.mul(X, Y), A), g.mul(g.mod(X, A), g.mod(Y, A)));
}

TEST(symbolic, unsolvable_mod_alignDown_changes_value) {
  SymGraph g;
  auto X = g.var();
  auto A = g.var();
  EXPECT_NE(g.mod(g.alignDown(X, A), A), g.mod(X, A));
}

TEST(symbolic, unsolvable_mod_alignUp_affects_sum) {
  SymGraph g;
  auto X = g.var();
  auto Y = g.var();
  auto A = g.var();
  // alignUp(X,A) ≡ 0 (mod A), so left becomes Y%A; right is (X+Y)%A
  EXPECT_NE(g.mod(g.add(g.alignUp(X, A), Y), A), g.mod(g.add(X, Y), A));
}

// ---- Mixed rounding/alignment combos ----

TEST(symbolic, unsolvable_align_combo_vs_ceilsum) {
  SymGraph g;
  auto X = g.var();
  auto Y = g.var();
  auto A = g.var();
  // LHS = ceil(X/A) + floor(Y/A); RHS = ceil((X+Y)/A)
  EXPECT_NE(g.div(g.add(g.alignUp(X, A), g.alignDown(Y, A)), A),
            g.cdiv(g.add(X, Y), A));
}

// ---- Multiplication vs division linearity (no overflow, but still not
// identities) ----

TEST(symbolic, unsolvable_div_product_not_linear_in_numerator) {
  SymGraph g;
  auto X = g.var();
  auto B = g.var();
  auto A = g.var();
  EXPECT_NE(g.div(g.mul(X, B), A), g.mul(g.div(X, A), B));
}

TEST(symbolic, unsolvable_mixed_floor_ceil_chain) {
  SymGraph g;
  auto X = g.var();
  auto A = g.var();
  auto B = g.var();
  // floor( ceil(X/A) / B )  vs  ceil( X / (A*B) )
  EXPECT_NE(g.div(g.cdiv(X, A), B), g.cdiv(X, g.mul(A, B)));
}

TEST(symbolic, unsolvable_div_idempotent_false) {
  SymGraph g;
  auto X = g.var();
  auto A = g.var();
  EXPECT_NE(g.div(g.div(X, A), A), g.div(X, A));
}

TEST(symbolic, unsolvable_cdiv_idempotent_false) {
  SymGraph g;
  auto X = g.var();
  auto A = g.var();
  EXPECT_NE(g.cdiv(g.cdiv(X, A), A), g.cdiv(X, A));
}

TEST(symbolic, unsolvable_cdiv_sum_vs_mixed) {
  SymGraph g;
  auto X = g.var();
  auto Y = g.var();
  auto A = g.var();
  // ceil((X+Y)/A) ≠ ceil(X/A) + floor(Y/A)
  EXPECT_NE(g.cdiv(g.add(X, Y), A), g.add(g.cdiv(X, A), g.div(Y, A)));
}

TEST(symbolic, unsolvable_div_diff_not_linear) {
  SymGraph g;
  auto X = g.var();
  auto Y = g.var();
  auto A = g.var();
  // From div(X-Y, A) we know X>=Y (no negative intermediate), still
  // non-identity:
  EXPECT_NE(g.div(g.sub(X, Y), A), g.sub(g.div(X, A), g.div(Y, A)));
}

TEST(symbolic, unsolvable_cdiv_diff_not_linear) {
  SymGraph g;
  auto X = g.var();
  auto Y = g.var();
  auto A = g.var();
  EXPECT_NE(g.cdiv(g.sub(X, Y), A), g.sub(g.cdiv(X, A), g.cdiv(Y, A)));
}

TEST(symbolic, unsolvable_half_stride_shift_not_plus_one) {
  SymGraph g;
  auto X = g.var();
  auto A = g.var();
  // floor((X + A) / (2A)) ≠ floor(X/(2A)) + 1
  EXPECT_NE(g.div(g.add(X, A), g.mul(2, A)), g.add(g.div(X, g.mul(2, A)), 1));
}

// ------------------------------ Alignment pitfalls
// ------------------------------

TEST(symbolic, unsolvable_alignUp_sum_non_additive) {
  SymGraph g;
  auto X = g.var();
  auto Y = g.var();
  auto A = g.var();
  EXPECT_NE(g.alignUp(g.add(X, Y), A), g.add(g.alignUp(X, A), g.alignUp(Y, A)));
}

TEST(symbolic, unsolvable_alignDown_sum_non_additive) {
  SymGraph g;
  auto X = g.var();
  auto Y = g.var();
  auto A = g.var();
  EXPECT_NE(g.alignDown(g.add(X, Y), A),
            g.add(g.alignDown(X, A), g.alignDown(Y, A)));
}

TEST(symbolic, unsolvable_alignUp_nested_vs_once) {
  SymGraph g;
  auto X = g.var();
  auto Y = g.var();
  auto A = g.var();
  // alignUp(alignUp(X,A)+Y, A) ≠ alignUp(X+Y, A) in general
  EXPECT_NE(g.alignUp(g.add(g.alignUp(X, A), Y), A), g.alignUp(g.add(X, Y), A));
}

TEST(symbolic, unsolvable_alignDown_nested_vs_once) {
  SymGraph g;
  auto X = g.var();
  auto Y = g.var();
  auto A = g.var();
  EXPECT_NE(g.alignDown(g.add(g.alignDown(X, A), Y), A),
            g.alignDown(g.add(X, Y), A));
}

// ----------------------------- Modulo (subtle non-equalities)
// -----------------------------

TEST(symbolic, unsolvable_mod_vs_scaled_addend) {
  SymGraph g;
  auto X = g.var();
  auto Y = g.var();
  auto A = g.var();
  // (X + Y*A) % A == X % A, BUT dropping the final %A on RHS is wrong:
  EXPECT_NE(g.mod(g.add(X, g.mul(Y, A)), A), g.mod(X, A)); // ok equality
  EXPECT_NE(g.mod(g.add(X, g.mul(Y, A)), A),
            X); // this should NOT be simplified to X
}

// ---------------------- Power mixed with integer rounding (not distributive)
// ----------------------

TEST(symbolic, unsolvable_pow_div_distribution_false) {
  SymGraph g;
  auto X = g.var();
  auto A = g.var();
  EXPECT_NE(g.div(g.pow(X, 2), A), g.pow(g.div(X, A), 2));
}

TEST(symbolic, unsolvable_pow_cdiv_distribution_false) {
  SymGraph g;
  auto X = g.var();
  auto A = g.var();
  EXPECT_NE(g.cdiv(g.pow(X, 2), A), g.pow(g.cdiv(X, A), 2));
}

TEST(symbolic, unsolvable_pow_mod_without_final_mod) {
  SymGraph g;
  auto X = g.var();
  auto A = g.var();
  EXPECT_NE(g.mod(g.pow(X, 2), A), g.pow(g.mod(X, A), 2));
}

// ------------------------------ Pool / cPool pitfalls
// ------------------------------

TEST(symbolic, unsolvable_pool_wrong_linearization_general_T) {
  SymGraph g;
  auto E = g.var();
  auto K = g.var();
  auto P = g.var();
  auto S = g.var();
  auto D = g.var();
  auto T = g.var();
  // pool(E+T) ≠ pool(E) + floor(T/S) for symbolic T (only true when T is a
  // multiple of S)
  EXPECT_NE(g.pool(g.add(E, T), K, P, S, D),
            g.add(g.pool(E, K, P, S, D), g.div(T, S)));
}

TEST(symbolic, unsolvable_pool_vs_cdiv_of_numerator) {
  SymGraph g;
  auto E = g.var();
  auto K = g.var();
  auto P = g.var();
  auto S = g.var();
  auto D = g.var();
  // pool = floor(num/S) + 1  vs  ceil(num/S)
  auto num = g.sub(g.add(E, g.mul(2, P)), g.add(g.mul(g.sub(K, 1), D), 1));
  EXPECT_NE(g.pool(E, K, P, S, D), g.cdiv(num, S));
}

TEST(symbolic, unsolvable_cpool_wrong_linearization_general_T) {
  SymGraph g;
  auto E = g.var();
  auto K = g.var();
  auto P = g.var();
  auto S = g.var();
  auto D = g.var();
  auto T = g.var();
  // cpool(E+T) ≠ cpool(E) + floor(T/S)  (only true when T is a multiple of S)
  EXPECT_NE(g.cpool(g.add(E, T), K, P, S, D),
            g.add(g.cpool(E, K, P, S, D), g.div(T, S)));
}

TEST(symbolic, unsolvable_cpool_vs_div_of_numerator_plus_one) {
  SymGraph g;
  auto E = g.var();
  auto K = g.var();
  auto P = g.var();
  auto S = g.var();
  auto D = g.var();
  auto num = g.sub(g.add(E, g.mul(2, P)), g.add(g.mul(g.sub(K, 1), D), 1));
  // cpool = ceil(num/S) + 1  vs  floor(num/S)+1
  EXPECT_NE(g.cpool(E, K, P, S, D), g.add(g.div(num, S), 1));
}

// ------------------------------ Mixed align/round interactions
// ------------------------------

TEST(symbolic, unsolvable_cdiv_of_alignDown_differs) {
  SymGraph g;
  auto X = g.var();
  auto A = g.var();
  // ceil( floor(X/A)*A / A ) == floor(X/A), which ≠ ceil(X/A) in general
  EXPECT_NE(g.cdiv(g.alignDown(X, A), A), g.cdiv(X, A));
}

// ----------------------- Pool composition / inversion myths
// -----------------------
TEST(symbolic, unsolvable_pool_compose_strides_naive_merge) {
  SymGraph g;
  auto E = g.var();
  auto K = g.var();
  auto P = g.var();
  auto S1 = g.var();
  auto S2 = g.var();
  auto D = g.var();
  // pool(pool(E, ..., S1), ..., S2)  ≠  pool(E, ..., S1*S2)
  EXPECT_NE(g.pool(g.pool(E, K, P, S1, D), K, P, S2, D),
            g.pool(E, K, P, g.mul(S1, S2), D));
}

TEST(symbolic, unsolvable_pool_scaled_input_not_equiv_stride1) {
  SymGraph g;
  auto E = g.var();
  auto K = g.var();
  auto P = g.var();
  auto S = g.var();
  auto D = g.var();
  // pool(E*S, ..., S)  ≠  pool(E, ..., 1)   (unless a special residue condition
  // holds)
  EXPECT_NE(g.pool(g.mul(E, S), K, P, S, D), g.pool(E, K, P, 1, D));
}

TEST(symbolic, unsolvable_pool_cannot_recover_numerator_exactly) {
  SymGraph g;
  auto E = g.var();
  auto K = g.var();
  auto P = g.var();
  auto S = g.var();
  auto D = g.var();
  auto eff = g.add(g.mul(g.sub(K, 1), D), 1);   // ((K-1)*D + 1)
  auto num = g.sub(g.add(E, g.mul(2, P)), eff); // E + 2P - eff
  // S * floor(num/S) == num  is false in general
  EXPECT_NE(g.mul(g.sub(g.pool(E, K, P, S, D), 1), S), num);
}

TEST(symbolic, unsolvable_pool_alignup_based_overcount) {
  SymGraph g;
  auto E = g.var();
  auto K = g.var();
  auto P = g.var();
  auto S = g.var();
  auto D = g.var();
  auto eff = g.add(g.mul(g.sub(K, 1), D), 1);
  auto num = g.sub(g.add(E, g.mul(2, P)), eff);
  // ceil(num/S)+1  ≠  floor(num/S)+1  generally
  EXPECT_NE(g.add(g.div(g.alignUp(num, S), S), 1), g.pool(E, K, P, S, D));
}

TEST(symbolic, unsolvable_pool_delta_not_cdiv_of_delta) {
  SymGraph g;
  auto E = g.var();
  auto T = g.var();
  auto K = g.var();
  auto P = g.var();
  auto S = g.var();
  auto D = g.var();
  // pool(E+T) - pool(E)  ≠  ceil(T/S)  in general
  EXPECT_NE(g.sub(g.pool(g.add(E, T), K, P, S, D), g.pool(E, K, P, S, D)),
            g.cdiv(T, S));
}

TEST(symbolic, unsolvable_pool_product_axis_mixing) {
  SymGraph g;
  auto W = g.var();
  auto H = g.var();
  auto K = g.var();
  auto P = g.var();
  auto S = g.var();
  auto D = g.var();
  // Pooling per-axis and multiplying counts is not equivalent to pooling total
  // elements.
  EXPECT_NE(g.mul(g.pool(W, K, P, S, D), g.pool(H, K, P, S, D)),
            g.pool(g.mul(W, H), K, P, S, D));
}

// ----------------------- Ceil/Floor with scale and sums (CNN-ish)
// -----------------------

TEST(symbolic, unsolvable_tiles_2D_not_equal_to_1D_over_area) {
  SymGraph g;
  auto W = g.var();
  auto H = g.var();
  auto Bx = g.var();
  auto By = g.var();
  // ceil(W/Bx)*ceil(H/By)  ≠  ceil(W*H/(Bx*By))
  EXPECT_NE(g.mul(g.cdiv(W, Bx), g.cdiv(H, By)),
            g.cdiv(g.mul(W, H), g.mul(Bx, By)));
}

TEST(symbolic, unsolvable_cdiv_not_linear_over_scale_then_sum) {
  SymGraph g;
  auto X = g.var();
  auto Y = g.var();
  auto S = g.var();
  // ceil((S*X + Y)/S)  ≠  S*ceil(X) + ceil(Y/S)   (note: ceil(X) is not
  // expressible; we show inequivalence vs ceil(X + Y/S))
  EXPECT_NE(g.cdiv(g.add(g.mul(S, X), Y), S), g.cdiv(g.add(X, g.div(Y, S)), 1));
}

// ----------------------- Alignment across axes / groups
// -----------------------

TEST(symbolic, unsolvable_alignUp_area_vs_axiswise) {
  SymGraph g;
  auto W = g.var();
  auto H = g.var();
  auto A = g.var();
  // alignUp(W*H, A)  ≠  alignUp(W, A) * alignUp(H, A)
  EXPECT_NE(g.alignUp(g.mul(W, H), A), g.mul(g.alignUp(W, A), g.alignUp(H, A)));
}

TEST(symbolic, unsolvable_alignUp_channels_times_groups_wrong) {
  SymGraph g;
  auto C = g.var();
  auto G = g.var();
  // alignUp(C, G) * G  ≠  alignUp(C*G, G)
  EXPECT_NE(g.mul(g.alignUp(C, G), G), g.alignUp(g.mul(C, G), G));
}

TEST(symbolic, unsolvable_alignUp_division_not_floor_tilecount) {
  SymGraph g;
  auto W = g.var();
  auto A = g.var();
  // ceil(W/A)  ≠  floor(W/A)
  EXPECT_NE(g.div(g.alignUp(W, A), A), g.div(W, A));
}

TEST(symbolic, unsolvable_floor_tilecount_times_tile_not_equal_alignUp) {
  SymGraph g;
  auto W = g.var();
  auto T = g.var();
  // floor(W/T)*T  ≠  alignUp(W, T)
  EXPECT_NE(g.mul(g.div(W, T), T), g.alignUp(W, T));
}

TEST(symbolic, unsolvable_down_then_align_vs_align_then_down) {
  SymGraph g;
  auto X = g.var();
  auto A = g.var();
  auto S = g.var();
  // floor( alignUp(X,S) / A )  ≠  floor( X / A ) in general
  EXPECT_NE(g.div(g.alignUp(X, S), A), g.div(X, A));
}

TEST(symbolic, unsolvable_alignUp_after_ceil_tiles_doesnt_match_alignDown) {
  SymGraph g;
  auto E = g.var();
  auto S = g.var();
  // alignDown( ceil(E/S)*S, S ) == alignUp(E,S)  ≠  alignDown(E,S)
  EXPECT_NE(g.alignDown(g.mul(g.cdiv(E, S), S), S), g.alignDown(E, S));
}

// ----------------------- “SAME” padding myths -----------------------

TEST(symbolic, unsolvable_same_padding_not_just_cdivE) {
  SymGraph g;
  auto E = g.var();
  auto K = g.var();
  auto S = g.var();
  auto D = g.var();
  // Using P = floor(((K-1)*D)/2) (often claimed for "SAME"):
  auto P = g.div(g.mul(g.sub(K, 1), D), 2);
  // pool(E,K,P,S,D)  ≠  ceil(E/S)   (not an identity for symbolic K,D)
  EXPECT_NE(g.pool(E, K, P, S, D), g.cdiv(E, S));
}

// ----------------------- Multi-stage “undo” myths -----------------------

TEST(symbolic, unsolvable_downsample_then_upsample_not_identity) {
  SymGraph g;
  auto E = g.var();
  auto K = g.var();
  auto P = g.var();
  auto S = g.var();
  auto D = g.var();
  auto U = g.var();
  // pool then scale back by U (nearest-neighbor style) is not identity:
  EXPECT_NE(g.mul(g.pool(E, K, P, S, D), U), E);
}

TEST(symbolic, unsolvable_stride_recover_with_align_not_identity) {
  SymGraph g;
  auto E = g.var();
  auto K = g.var();
  auto P = g.var();
  auto S = g.var();
  auto D = g.var();
  auto eff = g.add(g.mul(g.sub(K, 1), D), 1);
  auto num = g.sub(g.add(E, g.mul(2, P)), eff);
  // Aligning the “reconstructed numerator” from pooled count is not exact
  // recovery: alignUp( (pool-1)*S, S ) != num
  EXPECT_NE(g.alignUp(g.mul(g.sub(g.pool(E, K, P, S, D), 1), S), S), num);
}

// ----------------------- Splitting tensors then pooling
// -----------------------

TEST(symbolic, unsolvable_pool_concat_split_not_additive) {
  SymGraph g;
  auto E1 = g.var();
  auto E2 = g.var();
  auto K = g.var();
  auto P = g.var();
  auto S = g.var();
  auto D = g.var();
  // pool(E1+E2, ...)  ≠  pool(E1, ...) + pool(E2, ...)
  EXPECT_NE(g.pool(g.add(E1, E2), K, P, S, D),
            g.add(g.pool(E1, K, P, S, D), g.pool(E2, K, P, S, D)));
}

// ----------------------- “Ignoring kernel” or “off-by-one” myths
// -----------------------

TEST(symbolic, unsolvable_pool_not_just_cdiv_E_plus_padding) {
  SymGraph g;
  auto E = g.var();
  auto K = g.var();
  auto P = g.var();
  auto S = g.var();
  auto D = g.var();
  // Dropping the effective kernel size is wrong:
  EXPECT_NE(g.pool(E, K, P, S, D), g.cdiv(g.add(E, g.mul(2, P)), S));
}

TEST(symbolic, unsolvable_pool_off_by_one_variant_wrong) {
  SymGraph g;
  auto E = g.var();
  auto K = g.var();
  auto P = g.var();
  auto S = g.var();
  auto D = g.var();
  auto eff = g.add(g.mul(g.sub(K, 1), D), 1);
  auto num = g.sub(g.add(E, g.mul(2, P)), eff);
  // ceil((num + S - 1)/S) + 1  is not the same as floor(num/S)+1 in general.
  EXPECT_NE(g.add(g.cdiv(g.add(num, g.sub(S, 1)), S), 1),
            g.pool(E, K, P, S, D));
}

// New NE tests (all symbolic)
TEST(symbolic, unsolvable_two_stage_alignUp_vs_single_large) {
  SymGraph g;
  auto X = g.var();
  auto A = g.var();
  auto B = g.var();
  // Aligning to A then to B is not the same as aligning directly to A*B.
  EXPECT_NE(g.alignUp(g.alignUp(X, A), B), g.alignUp(X, g.mul(A, B)));
}

TEST(symbolic, unsolvable_two_stage_alignDown_vs_single_large) {
  SymGraph g;
  auto X = g.var();
  auto A = g.var();
  auto B = g.var();
  EXPECT_NE(g.alignDown(g.alignDown(X, A), B), g.alignDown(X, g.mul(A, B)));
}

TEST(symbolic, unsolvable_tilegrid_vs_area_tiles) {
  SymGraph g;
  auto W = g.var();
  auto H = g.var();
  auto T = g.var();
  // axiswise tiling vs area tiling
  EXPECT_NE(g.mul(g.cdiv(W, T), g.cdiv(H, T)),
            g.cdiv(g.mul(W, H), g.mul(T, T)));
}

TEST(symbolic, unsolvable_alignUp_sum_vs_sum_alignUp) {
  SymGraph g;
  auto X = g.var();
  auto Y = g.var();
  auto A = g.var();
  // Aligning a sum is not the sum of aligned parts.
  EXPECT_NE(g.alignUp(g.add(X, Y), A), g.add(g.alignUp(X, A), g.alignUp(Y, A)));
}

TEST(symbolic, unsolvable_cdiv_of_aligned_sum_vs_ceilsum) {
  SymGraph g;
  auto X = g.var();
  auto Y = g.var();
  auto A = g.var();
  // Left is exact (multiples of A), right is rounded sum.
  EXPECT_NE(g.div(g.add(g.alignUp(X, A), g.alignUp(Y, A)), A),
            g.cdiv(g.add(X, Y), A));
}

TEST(symbolic, unsolvable_pool_reconstruction_misses_remainder) {
  SymGraph g;
  auto E = g.var();
  auto K = g.var();
  auto P = g.var();
  auto S = g.var();
  auto D = g.var();
  auto eff = g.add(g.mul(g.sub(K, 1), D), 1);
  // "Invert" pool ignoring remainder: S*(pool-1) + eff - 2P != E in general
  EXPECT_NE(
      g.add(g.mul(g.sub(g.pool(E, K, P, S, D), 1), S), g.sub(eff, g.mul(2, P))),
      E);
}

TEST(symbolic, unsolvable_pool_compose_full_wrong_merge) {
  SymGraph g;
  auto E = g.var();
  auto K1 = g.var();
  auto P1 = g.var();
  auto S1 = g.var();
  auto D1 = g.var();
  auto K2 = g.var();
  auto P2 = g.var();
  auto S2 = g.var();
  auto D2 = g.var();
  // Merging two arbitrary pooling layers into one is not generally possible:
  EXPECT_NE(
      g.pool(g.pool(E, K1, P1, S1, D1), K2, P2, S2, D2),
      g.pool(E, g.mul(K1, K2), g.add(P1, P2), g.mul(S1, S2), g.mul(D1, D2)));
}

TEST(symbolic, unsolvable_cdiv_axiswise_vs_flatten) {
  SymGraph g;
  auto W = g.var();
  auto H = g.var();
  auto BX = g.var();
  auto BY = g.var();
  // tiles along each axis vs tiles on flattened dimension
  EXPECT_NE(g.mul(g.cdiv(W, BX), g.cdiv(H, BY)),
            g.cdiv(g.mul(W, H), g.mul(BX, BY)));
}

TEST(symbolic, unsolvable_alignUp_distance_not_always_one_tile) {
  SymGraph g;
  auto X = g.var();
  auto A = g.var();
  // Distance to next alignment can be 0..A-1; not always exactly one tile.
  EXPECT_NE(g.cdiv(g.sub(g.alignUp(X, A), X), A), g.resolve(1));
}

TEST(symbolic, unsolvable_pool_delta_vs_cdivT) {
  SymGraph g;
  auto E = g.var();
  auto T = g.var();
  auto K = g.var();
  auto P = g.var();
  auto S = g.var();
  auto D = g.var();
  // pool(E+T)-pool(E) is not ceil(T/S) in general.
  EXPECT_NE(g.sub(g.pool(g.add(E, T), K, P, S, D), g.pool(E, K, P, S, D)),
            g.cdiv(T, S));
}

// Two-stage pooling ≠ single pooled by product stride (ceil isn’t associative)
TEST(symbolic, unsolvable_pool_two_stage_vs_single) {
  SymGraph g;
  auto X = g.var(), S1 = g.var(), S2 = g.var();
  auto two = g.pool(g.pool(X, 1, 0, S1, 1), 1, 0, S2, 1); // ceil(ceil(X/S1)/S2)
  auto one = g.cdiv(X, g.mul(S1, S2));                    // ceil(X/(S1*S2))
  EXPECT_NE(NF(g, two), NF(g, one));
}

// -----------------------------------------------------------------------------
// AlignUp equivalences that require divisibility (but here it doesn't hold)
// -----------------------------------------------------------------------------

// If K ∤ A, then: alignUp(W*K, A)  ≠  K * alignUp(W, A/K)   (using trunc div)
TEST(symbolic, unsolvable_align_equivalence_requires_divisibility) {
  SymGraph g;
  auto W = g.var();
  int C = 16, B = 1, A = 66, K = C * B; // K=16, A/K = 4 (trunc), but K ∤ A
  auto lhs = g.alignUp(g.mul(W, K), A);
  auto rhs = g.mul(g.alignUp(W, A / K), K);
  EXPECT_NE(NF(g, lhs), NF(g, rhs));
}

// And the bytes-vs-pixels variant also fails without K | A
TEST(symbolic,
     unsolvable_align_equivalence_bytes_vs_pixels_requires_divisibility) {
  SymGraph g;
  auto W = g.var();
  int C = 16, B = 1, A = 66, K = C * B; // K=16, A/K=4 (trunc), but K ∤ A
  auto lhs = g.div(g.alignUp(g.mul(W, K), A), K);
  auto rhs = g.alignUp(W, A / K);
  EXPECT_NE(NF(g, lhs), NF(g, rhs));
}

// -----------------------------------------------------------------------------
// Mod pitfalls (no illicit distribution or over-cancellation)
// -----------------------------------------------------------------------------

// (X + Y) % Z  ≠  X%Z + Y%Z    (without another %Z)
TEST(symbolic, unsolvable_mod_no_add_distrib) {
  SymGraph g;
  auto X = g.var(), Y = g.var(), Z = g.var();
  auto lhs = g.mod(g.add(X, Y), Z);
  auto rhs = g.add(g.mod(X, Z), g.mod(Y, Z));
  EXPECT_NE(NF(g, lhs), NF(g, rhs));
}

// (A*B) % (A*C) is NOT guaranteed to be 0 unless B is a multiple of C
TEST(symbolic, unsolvable_mod_partial_product_not_zero) {
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var();
  auto e = g.mod(g.mul(A, B), g.mul(A, C));
  auto r = g.resolve(e);
  if (!r.isSymbolic()) {
    EXPECT_NE(0, r.constant()); // must not fold to 0
  } else {
    SUCCEED();
  }
}

// (X + Y*A) % A  ≠  X    (we must keep the final %A unless we know X%A==0)
TEST(symbolic, unsolvable_mod_vs_scaled_addend_drop_final_mod) {
  SymGraph g;
  auto X = g.var(), Y = g.var(), A = g.var();
  EXPECT_NE(NF(g, g.mod(g.add(X, g.mul(Y, A)), A)), NF(g, X));
}

// -----------------------------------------------------------------------------
// Division “algebra” that would be wrong without extra facts
// -----------------------------------------------------------------------------

// Dividing by a symbolic sum is NOT the same as dividing by a summand
TEST(symbolic, unsolvable_div_by_sum_vs_div_by_part) {
  SymGraph g;
  auto X = g.var(), A = g.var(), B = g.var();
  EXPECT_NE(NF(g, g.div(X, g.add(A, B))), NF(g, g.div(X, A)));
}

// (B*X + Y) / (k*B)  ≠  ((X + Y/B) / k)   unless we know B | Y
TEST(symbolic, unsolvable_wrong_split_mixed_terms) {
  SymGraph g;
  auto B = g.var(), X = g.var(), Y = g.var();
  auto lhs = g.div(g.add(g.mul(B, X), Y), g.mul(3, B));
  auto rhs = g.div(g.add(X, g.div(Y, B)), 3);
  EXPECT_NE(NF(g, lhs), NF(g, rhs));
}

// Ceil-div isn’t associative: ceil(ceil(X/A)/B) ≠ ceil(X/(A*B)) in general
TEST(symbolic, unsolvable_cdiv_two_stage_vs_single) {
  SymGraph g;
  auto X = g.var(), A = g.var(), B = g.var();
  auto two = g.cdiv(g.cdiv(X, A), B);
  auto one = g.cdiv(X, g.mul(A, B));
  EXPECT_NE(NF(g, two), NF(g, one));
}

// -----------------------------------------------------------------------------
// Alignment doesn’t distribute over addition
// -----------------------------------------------------------------------------

// alignUp(X + Y, A)  ≠  alignUp(X, A) + alignUp(Y, A)
TEST(symbolic, unsolvable_align_sum_not_distributive) {
  SymGraph g;
  auto X = g.var(), Y = g.var(), A = g.var();
  auto lhs = g.alignUp(g.add(X, Y), A);
  auto rhs = g.add(g.alignUp(X, A), g.alignUp(Y, A));
  EXPECT_NE(NF(g, lhs), NF(g, rhs));
}

// -----------------------------------------------------------------------------
//  A) Pool / ceil-div pitfalls (depth & compositions)
// -----------------------------------------------------------------------------

// ceil((X + D)/T)  ≠  ceil(X/T) + ceil(D/T)  (no linearity)
TEST(symbolic, unsolvable_pool_delta_vs_cdivT_deep) {
  SymGraph g;
  auto X = g.var(), T = g.var(), D = g.var();
  auto lhs = g.pool(g.add(X, D), 1, 0, T, 1);
  auto rhs = g.add(g.cdiv(X, T), g.cdiv(D, T));
  EXPECT_NE(NF(g, lhs), NF(g, rhs));
}

// Two-stage pooling vs single by product stride (ceil isn’t associative)
TEST(symbolic, unsolvable_pool_two_stage_vs_single_deep) {
  SymGraph g;
  auto X = g.var(), S1 = g.var(), S2 = g.var();
  auto two = g.pool(g.pool(X, 1, 0, S1, 1), 1, 0, S2, 1); // ceil(ceil(X/S1)/S2)
  auto one = g.cdiv(X, g.mul(S1, S2));                    // ceil(X/(S1*S2))
  EXPECT_NE(NF(g, two), NF(g, one));
}

// Shifting before pooling doesn’t add linearly either
TEST(symbolic, unsolvable_pool_shift_inside) {
  SymGraph g;
  auto X = g.var(), S = g.var();
  auto lhs = g.pool(g.add(X, g.sub(S, 1)), 1, 0, S, 1); // ceil((X+S-1)/S)
  auto rhs = g.add(g.cdiv(X, S), 1);
  EXPECT_NE(NF(g, lhs), NF(g, rhs));
}

// -----------------------------------------------------------------------------
//  B) AlignUp pitfalls (only safe under divisibility you don’t have here)
// -----------------------------------------------------------------------------

TEST(symbolic, unsolvable_align_equivalence_requires_divisibility_deep) {
  SymGraph g;
  auto W = g.var();
  int C = 16, B = 1, A = 66, K = C * B; // K=16, but K ∤ A
  auto lhs = g.alignUp(g.mul(W, K), A);
  auto rhs = g.mul(g.alignUp(W, A / K), K);
  EXPECT_NE(NF(g, lhs), NF(g, rhs));
}

TEST(symbolic,
     unsolvable_align_equivalence_bytes_vs_pixels_requires_divisibility_deep) {
  SymGraph g;
  auto W = g.var();
  int C = 16, B = 1, A = 66, K = C * B; // K=16, K ∤ A
  auto lhs = g.div(g.alignUp(g.mul(W, K), A), K);
  auto rhs = g.alignUp(W, A / K);
  EXPECT_NE(NF(g, lhs), NF(g, rhs));
}

// Align does NOT distribute over addition
TEST(symbolic, unsolvable_align_sum_not_distributive_deep) {
  SymGraph g;
  auto X = g.var(), Y = g.var(), A = g.var();
  auto lhs = g.alignUp(g.add(X, Y), A);
  auto rhs = g.add(g.alignUp(X, A), g.alignUp(Y, A));
  EXPECT_NE(NF(g, lhs), NF(g, rhs));
}

// AlignUp on a remainder is NOT the remainder itself (except at 0)
TEST(symbolic, unsolvable_align_of_mod_not_identity) {
  SymGraph g;
  auto X = g.var(), A = g.var();
  auto lhs = g.alignUp(g.mod(X, A), A);
  auto rhs = g.mod(X, A);
  EXPECT_NE(NF(g, lhs), NF(g, rhs));
}

// -----------------------------------------------------------------------------
//  C) Division: unsafe cancellations / wrong splittings in deeper graphs
// -----------------------------------------------------------------------------

// Mixed factors: you cannot cancel C inside a denom built from B
TEST(symbolic, unsolvable_cancel_wrong_symbol) {
  SymGraph g;
  auto B = g.var(), C = g.var(), X = g.var(), Y = g.var();
  auto lhs = g.div(g.add(g.mul(B, X), g.mul(C, Y)), g.mul(3, B));
  auto rhs = g.div(g.add(X, Y), 3); // would wrongly cancel C
  EXPECT_NE(NF(g, lhs), NF(g, rhs));
}

// You can’t split by sum in the denominator
TEST(symbolic, unsolvable_div_by_sum_vs_div_by_part_deep) {
  SymGraph g;
  auto X = g.var(), A = g.var(), B = g.var();
  EXPECT_NE(NF(g, g.div(X, g.add(A, B))), NF(g, g.div(X, A)));
}

// Cancelling only some terms then dividing others by B is unsafe if B∤Y
TEST(symbolic, unsolvable_wrong_split_mixed_terms_deep) {
  SymGraph g;
  auto B = g.var(), X = g.var(), Y = g.var();
  auto lhs = g.div(g.add(g.mul(B, X), Y), g.mul(3, B));
  auto rhs = g.div(g.add(X, g.div(Y, B)), 3);
  EXPECT_NE(NF(g, lhs), NF(g, rhs));
}

// Not associative across unrelated constants/symbols
TEST(symbolic, unsolvable_div_cascading_wrong_assoc) {
  SymGraph g;
  auto A = g.var(), B = g.var(), C = g.var(), X = g.var();
  auto lhs = g.div(g.mul(A, X), g.mul(B, C));
  auto rhs = g.div(g.div(X, B), C);
  EXPECT_NE(NF(g, lhs), NF(g, rhs));
}

// Deep composite: don’t over-reduce across unrelated regions
TEST(symbolic, unsolvable_deep_composite_mix) {
  SymGraph g;
  auto A = g.var(), B = g.var();
  std::int64_t k = 3;
  auto X = g.var(), Y = g.var(), U = g.var(), V = g.var();

  // E1 = ((B*X + B*Y + V) / (k*B))*A + ((A*U + V) % A)
  auto num1 = g.add(g.add(g.mul(B, X), g.mul(B, Y)), V);
  auto e1 = g.add(g.mul(g.div(num1, g.mul(k, B)), A),
                  g.mod(g.add(g.mul(A, U), V), A));

  // E2 = ((X + Y)/k)*A + (V % A)
  auto e2 = g.add(g.mul(g.div(g.add(X, Y), k), A), g.mod(V, A));

  EXPECT_NE(NF(g, e1), NF(g, e2));
}

// -----------------------------------------------------------------------------
//  D) Mod pitfalls (don’t over-distribute / over-cancel) with depth
// -----------------------------------------------------------------------------

// (X + Y) % Z  ≠  X%Z + Y%Z   (without final %Z)
TEST(symbolic, unsolvable_mod_no_add_distrib_deep) {
  SymGraph g;
  auto X = g.var(), Y = g.var(), Z = g.var();
  auto lhs = g.mod(g.add(X, Y), Z);
  auto rhs = g.add(g.mod(X, Z), g.mod(Y, Z));
  EXPECT_NE(NF(g, lhs), NF(g, rhs));
}

// Presence of both A and B does NOT force (A*X + B*Y) % (A*B) == 0
TEST(symbolic, unsolvable_mod_partial_product_not_zero_deep) {
  SymGraph g;
  auto A = g.var(), B = g.var(), X = g.var(), Y = g.var();
  auto e =
      g.mod(g.add(g.mul(A, B), g.add(g.mul(A, X), g.mul(B, Y))), g.mul(A, B));
  auto r = g.resolve(e);
  if (!r.isSymbolic()) {
    EXPECT_NE(0, r.constant()); // must not fold spuriously to 0
  } else {
    SUCCEED();
  }
}

// (A*X + Y) % A is NOT guaranteed 0 unless A | Y
TEST(symbolic, unsolvable_mod_mixed_term_not_zero) {
  SymGraph g;
  auto A = g.var(), X = g.var(), Y = g.var();
  auto e = g.mod(g.add(g.mul(A, X), Y), A);
  auto r = g.resolve(e);
  if (!r.isSymbolic()) {
    EXPECT_NE(0, r.constant());
  } else {
    SUCCEED();
  }
}

// Guard against wrong gcd-like simplifications:
// (2A*X + 3A*Y) / (4A)  ≠  (X + Y)/2
TEST(symbolic, unsolvable_div_wrong_gcd_reasoning) {
  SymGraph g;
  auto A = g.var(), X = g.var(), Y = g.var();
  auto lhs =
      g.div(g.add(g.mul(g.mul(2, A), X), g.mul(g.mul(3, A), Y)), g.mul(4, A));
  auto rhs = g.div(g.add(X, Y), 2);
  EXPECT_NE(NF(g, lhs), NF(g, rhs));
}

// After partial cancellation, mod by B must not vanish:
// ((B*X) / (k*B)) % B  ≠  0  (in general)
TEST(symbolic, unsolvable_mod_after_peel_not_zero) {
  SymGraph g;
  auto B = g.var(), X = g.var();
  auto q = g.div(g.mul(B, X), g.mul(3, B)); // -> floor(X/3)
  auto m = g.mod(q, B);
  auto r = g.resolve(m);
  if (!r.isSymbolic()) {
    EXPECT_NE(0, r.constant());
  } else {
    SUCCEED();
  }
}

TEST(symbolic, unsolvable_pool4_upsample4) {
  SymGraph g;
  auto W = g.var();

  auto alignment = 8;
  auto W_padded = g.add(W, g.sub(alignment, g.mod(W, alignment)));
  auto p0 = g.pool(W_padded, 2, 0, 2);
  auto p1 = g.pool(p0, 2, 0, 2);
  auto p2 = g.pool(p1, 2, 0, 2);
  auto p3 = g.pool(p2, 2, 0, 2);

  auto u3 = g.mul(p3, 2);

  EXPECT_NE(u3, p2);
}

TEST(symbolic, unsolvable_mod_sub_mod) {
  SymGraph g;
  auto W = g.var();

  Sym lhs = g.mod(g.sub(16, g.mod(W, 16)), 16);
  Sym rhs = g.mod(W, 16);

  ASSERT_TRUE(lhs.isSymbolic());
  ASSERT_TRUE(rhs.isSymbolic());
  //
  EXPECT_NE(lhs, rhs);
}
