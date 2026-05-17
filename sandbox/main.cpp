#include <fmt/format.h>

#include "denox/symbolic/SymGraph.hpp"

int main() {
  using namespace denox;

  SymGraph symGraph;

  Sym x = symGraph.var();

  Sym res = symGraph.mul(x, 2);


  memory::vector<Sym::symbol> vip = {x.sym(), res.sym()};
  auto [symir, remap] = symGraph.compile2(vip);

  Sym ir_x = remap[x];
  Sym ir_x2 = remap[res];

  assert(symir.varCount == 1);
  assert(ir_x.isSymbolic());
  assert(ir_x2.isSymbolic());

  Sym::symbol x_sid = ir_x.sym();
  Sym::symbol x2_sid = ir_x2.sym();

  memory::vector<Sym> dp;
  dp.reserve(symir.varCount + symir.ops.size());

  dp.emplace_back(x); // <- var

  for (const SymIROp &op : symir.ops) {
    switch (op.opcode) {
    case denox::SymIROpCode::Add_SS:
      dp.emplace_back(symGraph.add(dp[static_cast<size_t>(op.lhs)],
                                   dp[static_cast<size_t>(op.rhs)]));
      break;
    case denox::SymIROpCode::Add_SC:
      dp.emplace_back(symGraph.add(dp[static_cast<size_t>(op.lhs)], op.rhs));
      break;
    case denox::SymIROpCode::Sub_SS:
      dp.emplace_back(symGraph.sub(dp[static_cast<size_t>(op.lhs)],
                                   dp[static_cast<size_t>(op.rhs)]));
      break;
    case denox::SymIROpCode::Sub_SC:
      dp.emplace_back(symGraph.sub(dp[static_cast<size_t>(op.lhs)], op.rhs));
      break;
    case denox::SymIROpCode::Sub_CS:
      dp.emplace_back(symGraph.sub(op.lhs, dp[static_cast<size_t>(op.rhs)]));
      break;
    case denox::SymIROpCode::Mul_SS:
      dp.emplace_back(symGraph.mul(dp[static_cast<size_t>(op.lhs)],
                                   dp[static_cast<size_t>(op.rhs)]));
      break;
    case denox::SymIROpCode::Mul_SC:
      dp.emplace_back(symGraph.mul(dp[static_cast<size_t>(op.lhs)], op.rhs));
      break;
    case denox::SymIROpCode::Div_SS:
      dp.emplace_back(symGraph.div(dp[static_cast<size_t>(op.lhs)],
                                   dp[static_cast<size_t>(op.rhs)]));
      break;
    case denox::SymIROpCode::Div_SC:
      dp.emplace_back(symGraph.div(dp[static_cast<size_t>(op.lhs)], op.rhs));
      break;
    case denox::SymIROpCode::Div_CS:
      dp.emplace_back(symGraph.div(op.lhs, dp[static_cast<size_t>(op.rhs)]));
      break;
    case denox::SymIROpCode::Mod_SS:
      dp.emplace_back(symGraph.mod(dp[static_cast<size_t>(op.lhs)],
                                   dp[static_cast<size_t>(op.rhs)]));
      break;
    case denox::SymIROpCode::Mod_SC:
      dp.emplace_back(symGraph.mod(dp[static_cast<size_t>(op.lhs)], op.rhs));
      break;
    case denox::SymIROpCode::Mod_CS:
      dp.emplace_back(symGraph.mod(op.lhs, dp[static_cast<size_t>(op.rhs)]));
      break;
    case denox::SymIROpCode::Min_SS:
      dp.emplace_back(symGraph.min(dp[static_cast<size_t>(op.lhs)],
                                   dp[static_cast<size_t>(op.rhs)]));
      break;
    case denox::SymIROpCode::Min_SC:
      dp.emplace_back(symGraph.min(dp[static_cast<size_t>(op.lhs)], op.rhs));
      break;
    case denox::SymIROpCode::Max_SS:
      dp.emplace_back(symGraph.max(dp[static_cast<size_t>(op.lhs)],
                                   dp[static_cast<size_t>(op.rhs)]));
      break;
    case denox::SymIROpCode::Max_SC:
      dp.emplace_back(symGraph.max(dp[static_cast<size_t>(op.lhs)], op.rhs));
      break;
    }
  }


  assert(x == dp[x_sid]);

  assert(res == dp[x2_sid]);

  fmt::println("alive");


}
