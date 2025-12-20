#include "memory/container/vector.hpp"
#include "symbolic/SymGraph.hpp"
#include "symbolic/SymGraphEval.hpp"
#include <gtest/gtest.h>

using namespace denox::compiler;
using namespace denox;

static std::pair<std::int64_t, std::int64_t>
evalAndCompile(const SymGraph &graph, Sym X, std::int64_t x, Sym Y,
               std::int64_t y, Sym r) {

  std::vector<SymSpec> symSpecs;
  symSpecs.emplace_back(X.sym(), x);
  symSpecs.emplace_back(Y.sym(), y);
  auto eval = graph.eval(symSpecs);

  std::vector<Sym::symbol> symbols;
  symbols.push_back(r.sym());
  auto [ir, remap] = graph.compile(symbols);

  std::vector<std::int64_t> dp(ir.ops.size() + ir.varCount);
  dp[remap[X].sym()] = x;
  dp[remap[Y].sym()] = y;

  for (std::size_t pc = 0; pc < ir.ops.size(); ++pc) {
    std::uint64_t rx = pc + ir.varCount;
    const auto &op = ir.ops[pc];
    switch (op.opcode) {
    case SymIROpCode::Add_SS: {
      dp[rx] = dp[static_cast<std::size_t>(op.lhs)] +
               dp[static_cast<std::size_t>(op.rhs)];
      break;
    }
    case SymIROpCode::Add_SC: {
      dp[rx] = dp[static_cast<std::size_t>(op.lhs)] + op.rhs;
      break;
    }
    case SymIROpCode::Sub_SS: {
      dp[rx] = dp[static_cast<std::size_t>(op.lhs)] -
               dp[static_cast<std::size_t>(op.rhs)];
      break;
    }
    case SymIROpCode::Sub_SC: {
      dp[rx] = dp[static_cast<std::size_t>(op.lhs)] - op.rhs;
      break;
    }
    case SymIROpCode::Sub_CS: {
      dp[rx] = op.lhs - dp[static_cast<std::size_t>(op.rhs)];
      break;
    }
    case SymIROpCode::Mul_SS: {
      dp[rx] = dp[static_cast<std::size_t>(op.lhs)] *
               dp[static_cast<std::size_t>(op.rhs)];
      break;
    }
    case SymIROpCode::Mul_SC: {
      dp[rx] = dp[static_cast<std::size_t>(op.lhs)] * op.rhs;
      break;
    }
    case SymIROpCode::Div_SS: {
      dp[rx] = dp[static_cast<std::size_t>(op.lhs)] /
               dp[static_cast<std::size_t>(op.rhs)];
      break;
    }
    case SymIROpCode::Div_SC: {
      dp[rx] = dp[static_cast<std::size_t>(op.lhs)] / op.rhs;
      break;
    }
    case SymIROpCode::Div_CS: {
      dp[rx] = op.lhs / dp[static_cast<std::size_t>(op.rhs)];
      break;
    }
    case SymIROpCode::Mod_SS: {
      dp[rx] = dp[static_cast<std::size_t>(op.lhs)] %
               dp[static_cast<std::size_t>(op.rhs)];
      if (dp[rx] < 0) {
        dp[rx] += dp[static_cast<std::size_t>(op.rhs)];
      }
      break;
    }
    case SymIROpCode::Mod_SC: {
      dp[rx] = dp[static_cast<std::size_t>(op.lhs)] % op.rhs;
      if (dp[rx] < 0) {
        dp[rx] += op.rhs;
      }
      break;
    }
    case SymIROpCode::Mod_CS: {
      dp[rx] = op.lhs % dp[static_cast<std::size_t>(op.rhs)];
      if (dp[rx] < 0) {
        dp[rx] += dp[static_cast<std::size_t>(op.rhs)];
      }
      break;
    }
    case SymIROpCode::Min_SS: {
      dp[rx] = std::min(dp[static_cast<std::size_t>(op.lhs)],
                        dp[static_cast<std::size_t>(op.rhs)]);
      break;
    }
    case SymIROpCode::Min_SC: {
      dp[rx] = std::min(dp[static_cast<std::size_t>(op.lhs)], op.rhs);
      break;
    }
    case SymIROpCode::Max_SS: {
      dp[rx] = std::max(dp[static_cast<std::size_t>(op.lhs)],
                        dp[static_cast<std::size_t>(op.rhs)]);
      break;
    }
    case SymIROpCode::Max_SC: {
      dp[rx] = std::max(dp[static_cast<std::size_t>(op.lhs)], op.rhs);
      break;
    }
    }
  }
  return {*eval[r], dp[remap[r].sym()]};
}

TEST(symbolic, compile_add) {
  SymGraph graph;

  auto X = graph.var();
  auto Y = graph.var();
  auto r = graph.add(X, Y);

  auto [eval_r, comp_r] = evalAndCompile(graph, X, 11, Y, 7, r);

  EXPECT_EQ(eval_r, comp_r);
}

TEST(symbolic, compile_sub) {
  SymGraph graph;

  auto X = graph.var();
  auto Y = graph.var();
  auto r = graph.sub(X, Y);

  auto [eval_r, comp_r] = evalAndCompile(graph, X, 11, Y, 7, r);

  EXPECT_EQ(eval_r, comp_r);
}

TEST(symbolic, compile_mul) {
  SymGraph graph;

  auto X = graph.var();
  auto Y = graph.var();
  auto r = graph.max(X, Y);

  auto [eval_r, comp_r] = evalAndCompile(graph, X, 11, Y, 7, r);

  EXPECT_EQ(eval_r, comp_r);
}

TEST(symbolic, compile_div) {
  SymGraph graph;

  auto X = graph.var();
  auto Y = graph.var();
  auto r = graph.div(X, Y);

  auto [eval_r, comp_r] = evalAndCompile(graph, X, 11, Y, 7, r);

  EXPECT_EQ(eval_r, comp_r);
}

TEST(symbolic, compile_mod) {
  SymGraph graph;

  auto X = graph.var();
  auto Y = graph.var();
  auto r = graph.mod(X, Y);

  auto [eval_r, comp_r] = evalAndCompile(graph, X, 11, Y, 7, r);

  EXPECT_EQ(eval_r, comp_r);
}

TEST(symbolic, compile_min) {
  SymGraph graph;

  auto X = graph.var();
  auto Y = graph.var();
  auto r = graph.min(X, Y);

  auto [eval_r, comp_r] = evalAndCompile(graph, X, 11, Y, 7, r);

  EXPECT_EQ(eval_r, comp_r);
}

TEST(symbolic, compile_max) {
  SymGraph graph;

  auto X = graph.var();
  auto Y = graph.var();
  auto r = graph.max(X, Y);

  auto [eval_r, comp_r] = evalAndCompile(graph, X, 11, Y, 7, r);

  EXPECT_EQ(eval_r, comp_r);
}

TEST(symbolic, compile_pool) {
  SymGraph graph;

  auto X = graph.var();
  auto Y = graph.var();
  auto xp = graph.pool(X, 3, 1, 1, 1);
  auto yp = graph.pool(Y, 3, 1, 1, 1);
  auto r = graph.mul(graph.mul(xp, yp), 3 * 2);

  auto [eval_r, comp_r] = evalAndCompile(graph, X, 11, Y, 7, r);

  EXPECT_EQ(eval_r, comp_r);
}

TEST(symbolic, compile_cpool) {
  SymGraph graph;

  auto X = graph.var();
  auto Y = graph.var();
  auto xp = graph.cpool(X, 3, 1, 1, 1);
  auto yp = graph.cpool(Y, 3, 1, 1, 1);
  auto r = graph.mul(graph.mul(xp, yp), 3 * 2);

  auto [eval_r, comp_r] = evalAndCompile(graph, X, 11, Y, 7, r);

  EXPECT_EQ(eval_r, comp_r);
}

TEST(symbolic, compile_align_up) {
  SymGraph graph;

  auto X = graph.var();
  auto Y = graph.var();
  auto r = graph.alignUp(X, Y);

  auto [eval_r, comp_r] = evalAndCompile(graph, X, 11, Y, 7, r);

  EXPECT_EQ(eval_r, comp_r);
}

TEST(symbolic, compile_align_down) {
  SymGraph graph;

  auto X = graph.var();
  auto Y = graph.var();
  auto r = graph.alignDown(X, Y);

  auto [eval_r, comp_r] = evalAndCompile(graph, X, 11, Y, 7, r);

  EXPECT_EQ(eval_r, comp_r);
}

TEST(symbolic, compile_cdiv) {
  SymGraph graph;

  auto X = graph.var();
  auto Y = graph.var();
  auto r = graph.cdiv(X, Y);

  auto [eval_r, comp_r] = evalAndCompile(graph, X, 11, Y, 7, r);

  EXPECT_EQ(eval_r, comp_r);
}

TEST(symbolic, compile_mod_align) {
  SymGraph graph;

  auto _ = graph.var();

  auto X = graph.var();
  auto X_aligned = graph.mod(graph.sub(64, graph.mod(X, 64)), 64);
  auto r = X_aligned;

  auto [eval_r, comp_r] = evalAndCompile(graph, X, 11, _, 7, r);

  EXPECT_EQ(eval_r, comp_r);
}

TEST(symbolic, compile_nontrivial) {
  SymGraph graph;

  auto X = graph.var();
  auto Y = graph.var();

  auto X_aligned = graph.mod(graph.sub(64, graph.mod(X, 64)), 64);
  auto Y_aligned = graph.mod(graph.sub(64, graph.mod(Y, 64)), 64);

  auto X_conv0 = graph.pool(X_aligned, 3, 1, 1, 1);
  auto Y_conv0 = graph.pool(Y_aligned, 3, 1, 1, 1);

  auto X_pool0 = graph.pool(X_conv0, 2, 0, 2);
  auto Y_pool0 = graph.pool(Y_conv0, 2, 0, 2);

  auto X_up0 = graph.mul(X_pool0, 2);
  auto Y_up0 = graph.mul(Y_pool0, 2);

  auto r = graph.mul(X_up0, Y_up0, 3 * 2);

  auto [eval_r, comp_r] = evalAndCompile(graph, X, 1920, Y, 1080, r);

  EXPECT_EQ(eval_r, comp_r);
}
