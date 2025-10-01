#include "algorithm/binary_op_permutations.hpp"
#include "diag/invalid_state.hpp"
#include "diag/logging.hpp"
#include "diag/unreachable.hpp"
#include "memory/container/dynamic_bitset.hpp"
#include "memory/container/optional.hpp"
#include "memory/container/vector.hpp"
#include "memory/hypergraph/AdjGraph.hpp"
#include "memory/hypergraph/ConstGraph.hpp"
#include "symbolic/SymGraph.hpp"
#include "symbolic/SymIR.hpp"
#include <algorithm>
#include <utility>

namespace denox::compiler {

std::pair<SymIR, SymRemap>
SymGraph::compile(memory::span<const symbol> symbols) const {
  SymIR ir;
  memory::vector<Sym> remap(m_expressions.size());

  struct SymValue {
    // if null this means this is generated custom value.
    memory::optional<Sym::symbol> original;
  };

  struct SymOp {
    SymIROpCode opcode;
    std::int64_t constant = 0;
  };

  memory::dynamic_bitset dno(m_expressions.size(), false);
  for (Sym::symbol s : symbols) {
    dno[s] = true;
  }

  memory::AdjGraph<SymValue, SymOp> adjSupergraph;
  memory::vector<memory::NodeId> symbolToNodeMap(m_expressions.size());

  std::map<std::pair<Sym::symbol, Sym::symbol>, memory::NodeId> mulCache;

  for (Sym::symbol s = 0; s < m_expressions.size(); ++s) {
    const auto &expr = m_expressions[s];

    bool isAffine = expr.expr != ExprType::Identity &&
                    expr.expr != ExprType::NonAffine &&
                    expr.expr != ExprType::Const;

    if (isAffine) {
      Sym lhs = resolve(expr.lhs);
      Sym rhs = resolve(expr.rhs);
      memory::NodeId sid = adjSupergraph.addNode(SymValue{
          .original = s,
      });
      SymIROpCode opcode;
      opcode.lhsIsConstant = lhs.isConstant();
      opcode.rhsIsConstant = rhs.isConstant();
      switch (expr.expr) {
      case symbolic::details::ExprType::Div:
        opcode.op = SymIROpCode::OP_DIV;
        break;
      case symbolic::details::ExprType::Mod:
        opcode.op = SymIROpCode::OP_MOD;
        break;
      case symbolic::details::ExprType::Sub:
        opcode.op = SymIROpCode::OP_SUB;
        break;
      case symbolic::details::ExprType::Mul:
        opcode.op = SymIROpCode::OP_MUL;
        break;
      case symbolic::details::ExprType::Add:
        opcode.op = SymIROpCode::OP_ADD;
        break;
      case symbolic::details::ExprType::Min:
        DENOX_WARN("Iam skeptial if this is actually right");
        opcode.op = SymIROpCode::OP_MIN;
        break;
      case symbolic::details::ExprType::Max:
        DENOX_WARN("Iam skeptial if this is actually right");
        opcode.op = SymIROpCode::OP_MAX;
        break;
      case symbolic::details::ExprType::Const:
      case symbolic::details::ExprType::Identity:
      case symbolic::details::ExprType::NonAffine:
        diag::unreachable();
      }
      if (lhs.isConstant() && rhs.isConstant()) {
        diag::invalid_state();
      } else if (lhs.isConstant()) {
        assert(rhs.isSymbolic());
        memory::NodeId rid = symbolToNodeMap[rhs.sym()];
        adjSupergraph.addEdge(rid, sid, SymOp{opcode, lhs.constant()});
      } else if (expr.rhs.isConstant()) {
        assert(expr.lhs.isSymbolic());
        memory::NodeId lid = symbolToNodeMap[expr.lhs.sym()];
        adjSupergraph.addEdge(lid, sid, SymOp{opcode, expr.rhs.constant()});
      } else {
        assert(expr.lhs.isSymbolic());
        assert(expr.rhs.isSymbolic());
        memory::NodeId lhs = symbolToNodeMap[expr.lhs.sym()];
        memory::NodeId rhs = symbolToNodeMap[expr.rhs.sym()];
        adjSupergraph.addEdge(lhs, rhs, sid, SymOp{opcode});
      }
      break;
    } else {
      switch (expr.expr) {
      case symbolic::details::ExprType::Identity: {
        memory::NodeId sid = adjSupergraph.addNode(SymValue{
            .original = s,
        });
        symbolToNodeMap[s] = sid;
        break;
      }
      case symbolic::details::ExprType::NonAffine: {
        const auto &nonaffine = m_nonAffineCache.expressions[expr.lhs.sym()];
        switch (nonaffine.expr) {
        case symbolic::details::ExprType::Identity:
        case symbolic::details::ExprType::NonAffine:
        case symbolic::details::ExprType::Sub:
        case symbolic::details::ExprType::Add:
        case symbolic::details::ExprType::Const:
          diag::unreachable();
        case symbolic::details::ExprType::Div: {
          Sym lhs = resolve(nonaffine.symbols[0]);
          Sym rhs = resolve(nonaffine.symbols[1]);
          memory::NodeId sid = adjSupergraph.addNode(SymValue{
              .original = s,
          });
          symbolToNodeMap[s] = sid;
          SymIROpCode opcode;
          opcode.lhsIsConstant = lhs.isConstant();
          opcode.rhsIsConstant = rhs.isConstant();
          opcode.op = SymIROpCode::OP_DIV;
          if (lhs.isConstant() && rhs.isConstant()) {
            diag::invalid_state();
          } else if (lhs.isConstant()) {
            assert(rhs.isSymbolic());
            memory::NodeId rid = symbolToNodeMap[rhs.sym()];
            adjSupergraph.addEdge(rid, sid, SymOp{opcode, lhs.constant()});
          } else if (expr.rhs.isConstant()) {
            assert(expr.lhs.isSymbolic());
            memory::NodeId lid = symbolToNodeMap[expr.lhs.sym()];
            adjSupergraph.addEdge(lid, sid, SymOp{opcode, expr.rhs.constant()});
          } else {
            assert(expr.lhs.isSymbolic());
            assert(expr.rhs.isSymbolic());
            memory::NodeId lhs = symbolToNodeMap[expr.lhs.sym()];
            memory::NodeId rhs = symbolToNodeMap[expr.rhs.sym()];
            adjSupergraph.addEdge(lhs, rhs, sid, SymOp{opcode});
          }
          break;
        }
        case symbolic::details::ExprType::Mod: {
          Sym lhs = resolve(nonaffine.symbols[0]);
          Sym rhs = resolve(nonaffine.symbols[1]);
          memory::NodeId sid = adjSupergraph.addNode(SymValue{
              .original = s,
          });
          symbolToNodeMap[s] = sid;
          SymIROpCode opcode;
          opcode.lhsIsConstant = lhs.isConstant();
          opcode.rhsIsConstant = rhs.isConstant();
          opcode.op = SymIROpCode::OP_MOD;
          if (lhs.isConstant() && rhs.isConstant()) {
            diag::invalid_state();
          } else if (lhs.isConstant()) {
            assert(rhs.isSymbolic());
            memory::NodeId rid = symbolToNodeMap[rhs.sym()];
            adjSupergraph.addEdge(rid, sid, SymOp{opcode, lhs.constant()});
          } else if (expr.rhs.isConstant()) {
            assert(expr.lhs.isSymbolic());
            memory::NodeId lid = symbolToNodeMap[expr.lhs.sym()];
            adjSupergraph.addEdge(lid, sid, SymOp{opcode, expr.rhs.constant()});
          } else {
            assert(expr.lhs.isSymbolic());
            assert(expr.rhs.isSymbolic());
            memory::NodeId lhs = symbolToNodeMap[expr.lhs.sym()];
            memory::NodeId rhs = symbolToNodeMap[expr.rhs.sym()];
            adjSupergraph.addEdge(lhs, rhs, sid, SymOp{opcode});
          }
          break;
        }
        case symbolic::details::ExprType::Mul: {
          std::int64_t K = 1;
          memory::vector<Sym::symbol> factors;
          factors.reserve(nonaffine.symbols.size());
          for (const Sym &sf : nonaffine.symbols) {
            Sym r = resolve(sf);
            if (r.isConstant()) {
              K *= r.constant();
            } else {
              factors.push_back(r.sym());
            }
          }
          if (factors.empty()) {
            remap[s] = Sym::Const(K);
            break;
          }
          if (factors.size() == 1) {
            memory::NodeId sid = adjSupergraph.addNode(SymValue{
                .original = s,
            });
            symbolToNodeMap[s] = sid;
            const Sym::symbol only = factors[0];
            memory::NodeId src = symbolToNodeMap[only];

            SymIROpCode opcode;
            opcode.op = SymIROpCode::OP_MUL;
            opcode.lhsIsConstant = false;
            opcode.rhsIsConstant = true;
            adjSupergraph.addEdge(src, sid, SymOp{opcode, K});
            break;
          }
          // NOTE: No need to sort the factors because nonaffine.symbols
          // is guaranteed to be sorted! (INVARIANT)

          memory::NodeId factorProd = adjSupergraph.addNode(SymValue{
              .original = memory::nullopt,
          });

          std::uint32_t k = static_cast<std::uint32_t>(factors.size());
          memory::vector<algorithm::BinaryOpPermutation> perms =
              algorithm::binary_op_permutation(k);

          SymIROpCode opcode;
          opcode.lhsIsConstant = false;
          opcode.rhsIsConstant = false;
          opcode.op = SymIROpCode::OP_MUL;
          for (const auto &perm : perms) {
            memory::vector<memory::NodeId> intermediates(perm.ops.size());
            for (std::size_t o = 0; o < perm.ops.size(); ++o) {
              const auto &op = perm.ops[o];
              memory::NodeId lhs;
              if (op.lhsIntermediate) {
                lhs = intermediates[op.lhs];
              } else {
                lhs = symbolToNodeMap[factors[op.lhs]];
              }
              memory::NodeId rhs;
              if (op.rhsIntermediate) {
                rhs = intermediates[op.rhs];
              } else {
                rhs = symbolToNodeMap[factors[op.rhs]];
              }
              auto key = std::make_pair(std::min(lhs, rhs), std::max(lhs, rhs));
              if (o == perm.ops.size() - 1) {
                const auto it = mulCache.find(key);
                if (it == mulCache.end()) {
                  adjSupergraph.addEdge(lhs, rhs, factorProd, SymOp{opcode});
                  mulCache.insert(std::make_pair(key, factorProd));
                } else {
                  factorProd = it->second;
                  break;
                }
              } else {
                const auto it = mulCache.find(key);
                memory::NodeId dst;
                if (it == mulCache.end()) {
                  dst = adjSupergraph.addNode(SymValue{
                      .original = memory::nullopt, // <- intermediate value.
                  });
                  mulCache.insert(std::make_pair(key, dst));
                } else {
                  dst = it->second;
                }
                adjSupergraph.addEdge(lhs, rhs, dst, SymOp{opcode});
                intermediates[o] = dst;
              }
            }
          }
          if (K == 1) {
            adjSupergraph.get(factorProd).original = s;
            symbolToNodeMap[s] = factorProd;
          } else {
            memory::NodeId sid = adjSupergraph.addNode(SymValue{
                .original = s,
            });
            SymIROpCode opcode;
            opcode.lhsIsConstant = true;
            opcode.rhsIsConstant = false;
            opcode.op = SymIROpCode::OP_MUL;
            adjSupergraph.addEdge(factorProd, sid, SymOp{opcode, K});
          }
          break;
        }
        case symbolic::details::ExprType::Min: {
          std::int64_t K = 1;
          memory::vector<Sym::symbol> factors;
          factors.reserve(nonaffine.symbols.size());
          for (const Sym &sf : nonaffine.symbols) {
            Sym r = resolve(sf);
            if (r.isConstant()) {
              K *= r.constant();
            } else {
              factors.push_back(r.sym());
            }
          }
          if (factors.empty()) {
            remap[s] = Sym::Const(K);
            break;
          }
          if (factors.size() == 1) {
            memory::NodeId sid = adjSupergraph.addNode(SymValue{
                .original = s,
            });
            symbolToNodeMap[s] = sid;
            const Sym::symbol only = factors[0];
            memory::NodeId src = symbolToNodeMap[only];

            SymIROpCode opcode;
            opcode.op = SymIROpCode::OP_MIN;
            opcode.lhsIsConstant = false;
            opcode.rhsIsConstant = true;
            adjSupergraph.addEdge(src, sid, SymOp{opcode, K});
            break;
          }
          // NOTE: No need to sort the factors because nonaffine.symbols
          // is guaranteed to be sorted! (INVARIANT)

          memory::NodeId factorProd = adjSupergraph.addNode(SymValue{
              .original = memory::nullopt,
          });

          std::uint32_t k = static_cast<std::uint32_t>(factors.size());
          memory::vector<algorithm::BinaryOpPermutation> perms =
              algorithm::binary_op_permutation(k);

          SymIROpCode opcode;
          opcode.lhsIsConstant = false;
          opcode.rhsIsConstant = false;
          opcode.op = SymIROpCode::OP_MIN;
          for (const auto &perm : perms) {
            memory::vector<memory::NodeId> intermediates(perm.ops.size());
            for (std::size_t o = 0; o < perm.ops.size(); ++o) {
              const auto &op = perm.ops[o];
              memory::NodeId lhs;
              if (op.lhsIntermediate) {
                lhs = intermediates[op.lhs];
              } else {
                lhs = symbolToNodeMap[factors[op.lhs]];
              }
              memory::NodeId rhs;
              if (op.rhsIntermediate) {
                rhs = intermediates[op.rhs];
              } else {
                rhs = symbolToNodeMap[factors[op.rhs]];
              }
              auto key = std::make_pair(std::min(lhs, rhs), std::max(lhs, rhs));
              if (o == perm.ops.size() - 1) {
                const auto it = mulCache.find(key);
                if (it == mulCache.end()) {
                  adjSupergraph.addEdge(lhs, rhs, factorProd, SymOp{opcode});
                  mulCache.insert(std::make_pair(key, factorProd));
                } else {
                  factorProd = it->second;
                  break;
                }
              } else {
                const auto it = mulCache.find(key);
                memory::NodeId dst;
                if (it == mulCache.end()) {
                  dst = adjSupergraph.addNode(SymValue{
                      .original = memory::nullopt, // <- intermediate value.
                  });
                  mulCache.insert(std::make_pair(key, dst));
                } else {
                  dst = it->second;
                }
                adjSupergraph.addEdge(lhs, rhs, dst, SymOp{opcode});
                intermediates[o] = dst;
              }
            }
          }
          if (K == 1) {
            adjSupergraph.get(factorProd).original = s;
            symbolToNodeMap[s] = factorProd;
          } else {
            memory::NodeId sid = adjSupergraph.addNode(SymValue{
                .original = s,
            });
            SymIROpCode opcode;
            opcode.lhsIsConstant = true;
            opcode.rhsIsConstant = false;
            opcode.op = SymIROpCode::OP_MIN;
            adjSupergraph.addEdge(factorProd, sid, SymOp{opcode, K});
          }
          break;
        }
        case symbolic::details::ExprType::Max:
          std::int64_t K = 1;
          memory::vector<Sym::symbol> factors;
          factors.reserve(nonaffine.symbols.size());
          for (const Sym &sf : nonaffine.symbols) {
            Sym r = resolve(sf);
            if (r.isConstant()) {
              K *= r.constant();
            } else {
              factors.push_back(r.sym());
            }
          }
          if (factors.empty()) {
            remap[s] = Sym::Const(K);
            break;
          }
          if (factors.size() == 1) {
            memory::NodeId sid = adjSupergraph.addNode(SymValue{
                .original = s,
            });
            symbolToNodeMap[s] = sid;
            const Sym::symbol only = factors[0];
            memory::NodeId src = symbolToNodeMap[only];

            SymIROpCode opcode;
            opcode.op = SymIROpCode::OP_MAX;
            opcode.lhsIsConstant = false;
            opcode.rhsIsConstant = true;
            adjSupergraph.addEdge(src, sid, SymOp{opcode, K});
            break;
          }
          // NOTE: No need to sort the factors because nonaffine.symbols
          // is guaranteed to be sorted! (INVARIANT)

          memory::NodeId factorProd = adjSupergraph.addNode(SymValue{
              .original = memory::nullopt,
          });

          std::uint32_t k = static_cast<std::uint32_t>(factors.size());
          memory::vector<algorithm::BinaryOpPermutation> perms =
              algorithm::binary_op_permutation(k);

          SymIROpCode opcode;
          opcode.lhsIsConstant = false;
          opcode.rhsIsConstant = false;
          opcode.op = SymIROpCode::OP_MAX;
          for (const auto &perm : perms) {
            memory::vector<memory::NodeId> intermediates(perm.ops.size());
            for (std::size_t o = 0; o < perm.ops.size(); ++o) {
              const auto &op = perm.ops[o];
              memory::NodeId lhs;
              if (op.lhsIntermediate) {
                lhs = intermediates[op.lhs];
              } else {
                lhs = symbolToNodeMap[factors[op.lhs]];
              }
              memory::NodeId rhs;
              if (op.rhsIntermediate) {
                rhs = intermediates[op.rhs];
              } else {
                rhs = symbolToNodeMap[factors[op.rhs]];
              }
              auto key = std::make_pair(std::min(lhs, rhs), std::max(lhs, rhs));
              if (o == perm.ops.size() - 1) {
                const auto it = mulCache.find(key);
                if (it == mulCache.end()) {
                  adjSupergraph.addEdge(lhs, rhs, factorProd, SymOp{opcode});
                  mulCache.insert(std::make_pair(key, factorProd));
                } else {
                  factorProd = it->second;
                  break;
                }
              } else {
                const auto it = mulCache.find(key);
                memory::NodeId dst;
                if (it == mulCache.end()) {
                  dst = adjSupergraph.addNode(SymValue{
                      .original = memory::nullopt, // <- intermediate value.
                  });
                  mulCache.insert(std::make_pair(key, dst));
                } else {
                  dst = it->second;
                }
                adjSupergraph.addEdge(lhs, rhs, dst, SymOp{opcode});
                intermediates[o] = dst;
              }
            }
          }
          if (K == 1) {
            adjSupergraph.get(factorProd).original = s;
            symbolToNodeMap[s] = factorProd;
          } else {
            memory::NodeId sid = adjSupergraph.addNode(SymValue{
                .original = s,
            });
            SymIROpCode opcode;
            opcode.lhsIsConstant = true;
            opcode.rhsIsConstant = false;
            opcode.op = SymIROpCode::OP_MAX;
            adjSupergraph.addEdge(factorProd, sid, SymOp{opcode, K});
          }
          break;
        }
        break;
      }
      case symbolic::details::ExprType::Const: {
        assert(expr.lhs.isConstant());
        remap[s] = Sym::Const(expr.lhs.constant());
        break;
      }
      case symbolic::details::ExprType::Div:
      case symbolic::details::ExprType::Mod:
      case symbolic::details::ExprType::Sub:
      case symbolic::details::ExprType::Mul:
      case symbolic::details::ExprType::Add:
      case symbolic::details::ExprType::Min:
      case symbolic::details::ExprType::Max:
        diag::unreachable();
        break;
      }
    }
  }

  memory::ConstGraph<SymValue, SymOp> supergraph{adjSupergraph};

  return std::make_pair(ir, SymRemap(remap));
}

} // namespace denox::compiler
