#include "denox/symbolic/SymIR.hpp"
#include "denox/diag/unreachable.hpp"
#include <limits>

denox::SymIREval denox::SymIR::eval(memory::span<const SymSpec> specs) const {
  memory::vector<int64_t> dp(varCount + ops.size());

  for (const SymSpec &spec : specs) {
    if (spec.symbol >= varCount) {
      continue;
    }
    dp[spec.symbol] = spec.value;
  }

  for (size_t pc = 0; pc < ops.size(); ++pc) {
    size_t sid = pc + varCount;
    const auto &op = ops[pc];

    switch (op.opcode) {
    case SymIROpCode::Add_SS:
      dp[sid] =
          dp[static_cast<size_t>(op.lhs)] + dp[static_cast<size_t>(op.rhs)];
      break;
    case SymIROpCode::Add_SC:
      dp[sid] = dp[static_cast<size_t>(op.lhs)] + op.rhs;
      break;
    case SymIROpCode::Sub_SS:
      dp[sid] =
          dp[static_cast<size_t>(op.lhs)] - dp[static_cast<size_t>(op.rhs)];
      break;
    case SymIROpCode::Sub_SC:
      dp[sid] = dp[static_cast<size_t>(op.lhs)] - op.rhs;
      break;
    case SymIROpCode::Sub_CS:
      dp[sid] = op.lhs - dp[static_cast<size_t>(op.rhs)];
      break;
    case SymIROpCode::Mul_SS:
      dp[sid] =
          dp[static_cast<size_t>(op.lhs)] * dp[static_cast<size_t>(op.rhs)];
      break;
    case SymIROpCode::Mul_SC:
      dp[sid] = dp[static_cast<size_t>(op.lhs)] * op.rhs;
      break;
    case SymIROpCode::Div_SS:
      dp[sid] =
          dp[static_cast<size_t>(op.lhs)] / dp[static_cast<size_t>(op.rhs)];
      break;
    case SymIROpCode::Div_SC:
      dp[sid] = dp[static_cast<size_t>(op.lhs)] / op.rhs;
      break;
    case SymIROpCode::Div_CS:
      dp[sid] = op.lhs / dp[static_cast<size_t>(op.rhs)];
      break;
    case SymIROpCode::Mod_SS: {
      dp[sid] =
          dp[static_cast<size_t>(op.lhs)] % dp[static_cast<size_t>(op.rhs)];
      if (dp[sid] < 0) {
        dp[sid] += dp[static_cast<size_t>(op.rhs)];
      }
      break;
    }
    case SymIROpCode::Mod_SC: {
      dp[sid] = dp[static_cast<size_t>(op.lhs)] % op.rhs;
      if (dp[sid] < 0) {
        dp[sid] += op.rhs;
      }
      break;
    }
    case SymIROpCode::Mod_CS: {
      dp[sid] = op.lhs % dp[static_cast<size_t>(op.rhs)];
      if (dp[sid] < 0) {
        dp[sid] += dp[static_cast<size_t>(op.rhs)];
      }
      break;
    }
    case SymIROpCode::Min_SS:
      dp[sid] = std::min(dp[static_cast<size_t>(op.lhs)],
                         dp[static_cast<size_t>(op.rhs)]);
      break;
    case SymIROpCode::Min_SC:
      dp[sid] = std::min(dp[static_cast<size_t>(op.lhs)], op.rhs);
      break;
    case SymIROpCode::Max_SS:
      dp[sid] = std::max(dp[static_cast<size_t>(op.lhs)],
                         dp[static_cast<size_t>(op.rhs)]);
      break;
    case SymIROpCode::Max_SC:
      dp[sid] = std::max(dp[static_cast<size_t>(op.lhs)], op.rhs);
      break;
    default:
      diag::unreachable();
    }
  }

  return denox::SymIREval(std::move(dp));
}
