#include "algorithm/binary_op_permutations.hpp"
#include "algorithm/shortest_dag_hyperpath.hpp"
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
#include <fmt/base.h>
#include <spirv-tools/libspirv.h>
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

  fmt::println("expr : {}", m_expressions.size());
  memory::dynamic_bitset dno(m_expressions.size(), false);
  for (Sym::symbol s : symbols) {
    dno[s] = true;
  }

  using weight_type = std::int64_t;

  static constexpr weight_type ADD_WEIGHT = 0;
  static constexpr weight_type SUB_WEIGHT = 0;
  static constexpr weight_type MUL_WEIGHT = 0;
  static constexpr weight_type DIV_WEIGHT = 0;
  static constexpr weight_type MOD_WEIGHT = 0;
  static constexpr weight_type MIN_WEIGHT = 0;
  static constexpr weight_type MAX_WEIGHT = 0;

  memory::AdjGraph<SymValue, SymOp, weight_type> adjSupergraph;
  memory::vector<memory::NodeId> symbolToNodeMap(m_expressions.size());

  std::map<std::pair<Sym::symbol, Sym::symbol>, memory::NodeId> mulCache;
  std::map<std::pair<Sym::symbol, Sym::symbol>, memory::NodeId> minCache;
  std::map<std::pair<Sym::symbol, Sym::symbol>, memory::NodeId> maxCache;

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
      symbolToNodeMap[s] = sid;
      SymIROpCode opcode;
      opcode.lhsIsConstant = lhs.isConstant();
      opcode.rhsIsConstant = rhs.isConstant();
      weight_type weight;
      switch (expr.expr) {
      case symbolic::details::ExprType::Div:
        opcode.op = SymIROpCode::OP_DIV;
        weight = DIV_WEIGHT;
        break;
      case symbolic::details::ExprType::Mod:
        opcode.op = SymIROpCode::OP_MOD;
        weight = MOD_WEIGHT;
        break;
      case symbolic::details::ExprType::Sub:
        opcode.op = SymIROpCode::OP_SUB;
        weight = SUB_WEIGHT;
        break;
      case symbolic::details::ExprType::Mul:
        opcode.op = SymIROpCode::OP_MUL;
        weight = MUL_WEIGHT;
        break;
      case symbolic::details::ExprType::Add:
        opcode.op = SymIROpCode::OP_ADD;
        weight = ADD_WEIGHT;
        break;
      case symbolic::details::ExprType::Min:
        DENOX_WARN("Iam skeptial if this is actually right");
        opcode.op = SymIROpCode::OP_MIN;
        weight = MIN_WEIGHT;
        break;
      case symbolic::details::ExprType::Max:
        DENOX_WARN("Iam skeptial if this is actually right");
        opcode.op = SymIROpCode::OP_MAX;
        weight = MAX_WEIGHT;
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
        assert(rid);
        adjSupergraph.addEdge(rid, sid, SymOp{opcode, lhs.constant()}, weight);
      } else if (expr.rhs.isConstant()) {
        assert(expr.lhs.isSymbolic());
        memory::NodeId lid = symbolToNodeMap[expr.lhs.sym()];
        assert(lid);
        adjSupergraph.addEdge(lid, sid, SymOp{opcode, expr.rhs.constant()},
                              weight);
      } else {
        assert(expr.lhs.isSymbolic());
        assert(expr.rhs.isSymbolic());
        memory::NodeId lhs = symbolToNodeMap[expr.lhs.sym()];
        memory::NodeId rhs = symbolToNodeMap[expr.rhs.sym()];
        assert(lhs);
        assert(rhs);
        adjSupergraph.addEdge(lhs, rhs, sid, SymOp{opcode}, weight);
      }
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
            adjSupergraph.addEdge(rid, sid, SymOp{opcode, lhs.constant()},
                                  DIV_WEIGHT);
          } else if (expr.rhs.isConstant()) {
            assert(expr.lhs.isSymbolic());
            memory::NodeId lid = symbolToNodeMap[expr.lhs.sym()];
            adjSupergraph.addEdge(lid, sid, SymOp{opcode, expr.rhs.constant()},
                                  DIV_WEIGHT);
          } else {
            assert(expr.lhs.isSymbolic());
            assert(expr.rhs.isSymbolic());
            memory::NodeId lhs = symbolToNodeMap[expr.lhs.sym()];
            memory::NodeId rhs = symbolToNodeMap[expr.rhs.sym()];
            adjSupergraph.addEdge(lhs, rhs, sid, SymOp{opcode}, DIV_WEIGHT);
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
            adjSupergraph.addEdge(rid, sid, SymOp{opcode, lhs.constant()},
                                  MOD_WEIGHT);
          } else if (expr.rhs.isConstant()) {
            assert(expr.lhs.isSymbolic());
            memory::NodeId lid = symbolToNodeMap[expr.lhs.sym()];
            adjSupergraph.addEdge(lid, sid, SymOp{opcode, expr.rhs.constant()},
                                  MOD_WEIGHT);
          } else {
            assert(expr.lhs.isSymbolic());
            assert(expr.rhs.isSymbolic());
            memory::NodeId lhs = symbolToNodeMap[expr.lhs.sym()];
            memory::NodeId rhs = symbolToNodeMap[expr.rhs.sym()];
            adjSupergraph.addEdge(lhs, rhs, sid, SymOp{opcode}, MOD_WEIGHT);
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
            adjSupergraph.addEdge(src, sid, SymOp{opcode, K}, MUL_WEIGHT);
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
              auto key = std::make_pair(std::min(*lhs, *rhs), std::max(*lhs, *rhs));
              if (o == perm.ops.size() - 1) {
                const auto it = mulCache.find(key);
                if (it == mulCache.end()) {
                  adjSupergraph.addEdge(lhs, rhs, factorProd, SymOp{opcode},
                                        MUL_WEIGHT);
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
                adjSupergraph.addEdge(lhs, rhs, dst, SymOp{opcode}, MUL_WEIGHT);
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
            adjSupergraph.addEdge(factorProd, sid, SymOp{opcode, K},
                                  MUL_WEIGHT);
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
            adjSupergraph.addEdge(src, sid, SymOp{opcode, K}, MUL_WEIGHT);
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
              auto key = std::make_pair(std::min(*lhs, *rhs), std::max(*lhs, *rhs));
              if (o == perm.ops.size() - 1) {
                const auto it = minCache.find(key);
                if (it == minCache.end()) {
                  adjSupergraph.addEdge(lhs, rhs, factorProd, SymOp{opcode},
                                        MUL_WEIGHT);
                  minCache.insert(std::make_pair(key, factorProd));
                } else {
                  factorProd = it->second;
                  break;
                }
              } else {
                const auto it = minCache.find(key);
                memory::NodeId dst;
                if (it == minCache.end()) {
                  dst = adjSupergraph.addNode(SymValue{
                      .original = memory::nullopt, // <- intermediate value.
                  });
                  minCache.insert(std::make_pair(key, dst));
                } else {
                  dst = it->second;
                }
                adjSupergraph.addEdge(lhs, rhs, dst, SymOp{opcode}, MUL_WEIGHT);
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
            adjSupergraph.addEdge(factorProd, sid, SymOp{opcode, K},
                                  MUL_WEIGHT);
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
            adjSupergraph.addEdge(src, sid, SymOp{opcode, K}, MUL_WEIGHT);
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
              auto key = std::make_pair(std::min(*lhs, *rhs), std::max(*lhs, *rhs));
              if (o == perm.ops.size() - 1) {
                const auto it = maxCache.find(key);
                if (it == maxCache.end()) {
                  adjSupergraph.addEdge(lhs, rhs, factorProd, SymOp{opcode},
                                        MUL_WEIGHT);
                  maxCache.insert(std::make_pair(key, factorProd));
                } else {
                  factorProd = it->second;
                  break;
                }
              } else {
                const auto it = maxCache.find(key);
                memory::NodeId dst;
                if (it == maxCache.end()) {
                  dst = adjSupergraph.addNode(SymValue{
                      .original = memory::nullopt, // <- intermediate value.
                  });
                  maxCache.insert(std::make_pair(key, dst));
                } else {
                  dst = it->second;
                }
                adjSupergraph.addEdge(lhs, rhs, dst, SymOp{opcode}, MUL_WEIGHT);
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
            adjSupergraph.addEdge(factorProd, sid, SymOp{opcode, K},
                                  MUL_WEIGHT);
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

  debugDump();
  assert(adjSupergraph.nodeCount() >= m_expressions.size());
  memory::ConstGraph<SymValue, SymOp, weight_type> supergraph{adjSupergraph};
  memory::vector<memory::NodeId> vars;
  memory::vector<memory::NodeId> results;
  // Collect variables.
  for (std::size_t n = 0; n < supergraph.nodeCount(); ++n) {
    memory::NodeId nid{n};
    auto node = supergraph.get(nid);
    if (!node.original.has_value()) {
      continue;
    }
    Sym::symbol original = *node.original;
    if (dno[original]) {
      fmt::println("result symbol {}", original);
      results.push_back(nid);
    }
    if (m_expressions[original].expr == ExprType::Identity) {
      vars.push_back(nid);
    }
  }
  fmt::println("edges: {}", supergraph.edgeCount());
  fmt::println("nodes: {}", supergraph.nodeCount());
  fmt::println("vars: {}", vars.size());
  fmt::println("results: {}", results.size());
  fmt::println("symbols: {}", symbols.size());

  auto hyperpath = algorithm::shortest_dag_hyperpath(supergraph, vars, results);
  assert(hyperpath.has_value());
  fmt::println("len: {}", hyperpath->size());

  for (std::size_t e = 0; e < hyperpath->size(); ++e) {
    memory::EdgeId eid{e};
    auto op = supergraph.get(eid);

    auto dst = supergraph.dst(eid);
    fmt::print("[{}] = ", static_cast<std::uint64_t>(dst));

    if (op.opcode.lhsIsConstant) {
      fmt::print("{}", op.constant);
    } else {
      auto srcs = supergraph.src(eid);
      assert(srcs.size() == 1);
      memory::NodeId nid = srcs[0];
      fmt::print("[{}]", static_cast<std::uint64_t>(nid));
    }
    switch (op.opcode.op) {
      case SymIROpCode::OP_ADD:
        fmt::print(" + ");
        break;
      case SymIROpCode::OP_SUB:
        fmt::print(" + ");
        break;
      case SymIROpCode::OP_MUL:
        fmt::print(" * ");
        break;
      case SymIROpCode::OP_DIV:
        fmt::print(" / ");
        break;
      case SymIROpCode::OP_MOD:
        fmt::print(" % ");
        break;
      case SymIROpCode::OP_MIN:
        fmt::print(" min* ");
        break;
      case SymIROpCode::OP_MAX:
        fmt::print(" max* ");
        break;
    }

    if (op.opcode.rhsIsConstant) {
      fmt::println("{}", op.constant);
    } else {
      auto srcs = supergraph.src(eid);
      if (srcs.size() == 1) {
        fmt::println("[{}]", static_cast<std::uint64_t>(srcs[0]));
      } else {
        fmt::println("[{}]", static_cast<std::uint64_t>(srcs[1]));
      }
    }
  }

  return std::make_pair(ir, SymRemap(remap));
}

} // namespace denox::compiler
