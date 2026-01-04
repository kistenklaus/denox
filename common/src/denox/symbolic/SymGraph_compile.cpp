#include "denox/algorithm/binary_op_permutations.hpp"
#include "denox/algorithm/shortest_dag_hyperpath.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/diag/unreachable.hpp"
#include "denox/memory/container/dynamic_bitset.hpp"
#include "denox/memory/container/optional.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/memory/hypergraph/AdjGraph.hpp"
#include "denox/memory/hypergraph/ConstGraph.hpp"
#include "denox/symbolic/SymGraph.hpp"
#include "denox/symbolic/SymIR.hpp"
#include <algorithm>
#include <map>
#include <utility>

namespace denox {

std::pair<SymIR, SymRemap>
SymGraph::compile(memory::span<const symbol> symbols) const {
  SymIR ir;
  memory::vector<memory::optional<Sym>> remap(m_expressions.size(),
                                              memory::nullopt);

  struct SymValue {
    // if null this means this is intermediate unmapped value.
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

  using weight_type = float;

  static constexpr weight_type ADD_SS_WEIGHT = 1.0f;
  static constexpr weight_type ADD_SC_WEIGHT = 1.0f;
  static constexpr weight_type SUB_SS_WEIGHT = 1.0f;
  static constexpr weight_type SUB_SC_WEIGHT = 1.0f;
  static constexpr weight_type SUB_CS_WEIGHT = 1.0f;
  static constexpr weight_type MUL_SS_WEIGHT = 1.0f;
  static constexpr weight_type MUL_SC_WEIGHT = 1.0f;
  static constexpr weight_type DIV_SS_WEIGHT = 1.0f;
  static constexpr weight_type DIV_SC_WEIGHT = 1.0f;
  static constexpr weight_type DIV_CS_WEIGHT = 1.0f;
  static constexpr weight_type MOD_SS_WEIGHT = 1.0f;
  static constexpr weight_type MOD_SC_WEIGHT = 1.0f;
  static constexpr weight_type MOD_CS_WEIGHT = 1.0f;
  static constexpr weight_type MIN_SS_WEIGHT = 1.0f;
  static constexpr weight_type MIN_SC_WEIGHT = 1.0f;
  static constexpr weight_type MAX_SS_WEIGHT = 1.0f;
  static constexpr weight_type MAX_SC_WEIGHT = 1.0f;

  memory::AdjGraph<SymValue, SymOp, weight_type> adjSupergraph;
  memory::vector<memory::NodeId> symbolToNodeMap(m_expressions.size());

  std::map<std::pair<Sym::symbol, Sym::symbol>, memory::NodeId> mulCache;
  std::map<std::pair<Sym::symbol, Sym::symbol>, memory::NodeId> minCache;
  std::map<std::pair<Sym::symbol, Sym::symbol>, memory::NodeId> maxCache;

  for (Sym::symbol s = 0; s < m_expressions.size(); ++s) {
    const auto expr = m_expressions[s];

    bool isAffine = expr.expr != ExprType::Identity &&
                    expr.expr != ExprType::NonAffine &&
                    expr.expr != ExprType::Const;

    if (isAffine) {
      Sym lhs = resolve(expr.lhs);
      Sym rhs = resolve(expr.rhs);

      // if (lhs.isConstant() && rhs.isConstant()) {
      // }
      Sym::value_type constant = 0;
      memory::optional<SymIROpCode> opcode = memory::nullopt;
      weight_type weight = 0;
      switch (expr.expr) {
      case symbolic::details::ExprType::Div:
        if (lhs.isConstant() && rhs.isConstant()) {
          constant = lhs.constant() / rhs.constant();
        } else if (lhs.isConstant()) {
          opcode = SymIROpCode::Div_CS;
          constant = lhs.constant();
          weight = DIV_CS_WEIGHT;
        } else if (rhs.isConstant()) {
          opcode = SymIROpCode::Div_SC;
          constant = rhs.constant();
          weight = DIV_SC_WEIGHT;
        } else {
          opcode = SymIROpCode::Div_SS;
          weight = DIV_SS_WEIGHT;
        }
        break;
      case symbolic::details::ExprType::Mod:
        if (lhs.isConstant() && rhs.isConstant()) {
          constant = emod(lhs.constant(), rhs.constant());
        } else if (lhs.isConstant()) {
          opcode = SymIROpCode::Div_CS;
          constant = lhs.constant();
          weight = MOD_CS_WEIGHT;
        } else if (rhs.isConstant()) {
          opcode = SymIROpCode::Div_SC;
          constant = rhs.constant();
          weight = MOD_SC_WEIGHT;
        } else {
          opcode = SymIROpCode::Div_SS;
          weight = MOD_SS_WEIGHT;
        }
        break;
      case symbolic::details::ExprType::Sub:
        if (lhs.isConstant() && rhs.isConstant()) {
          constant = lhs.constant() - rhs.constant();
        } else if (lhs.isConstant()) {
          opcode = SymIROpCode::Sub_CS;
          constant = lhs.constant();
          weight = SUB_CS_WEIGHT;
        } else if (rhs.isConstant()) {
          opcode = SymIROpCode::Sub_SC;
          constant = rhs.constant();
          weight = SUB_SC_WEIGHT;
        } else {
          opcode = SymIROpCode::Sub_SS;
          weight = SUB_SS_WEIGHT;
        }
        break;
      case symbolic::details::ExprType::Mul:
        if (lhs.isConstant() && rhs.isConstant()) {
          constant = lhs.constant() * rhs.constant();
        } else if (lhs.isSymbolic() && rhs.isSymbolic()) {
          opcode = SymIROpCode::Mul_SS;
          weight = MUL_SS_WEIGHT;
        } else {
          opcode = SymIROpCode::Mul_SC;
          constant = rhs.isConstant() ? rhs.constant() : lhs.constant();
          weight = MUL_SC_WEIGHT;
        }
        break;
      case symbolic::details::ExprType::Add:
        if (lhs.isConstant() && rhs.isConstant()) {
          constant = lhs.constant() * rhs.constant();
        } else if (lhs.isSymbolic() && rhs.isSymbolic()) {
          opcode = SymIROpCode::Add_SS;
          weight = ADD_SS_WEIGHT;
        } else {
          opcode = SymIROpCode::Add_SC;
          constant = rhs.isConstant() ? rhs.constant() : lhs.constant();
          weight = ADD_SC_WEIGHT;
        }
        break;
      case symbolic::details::ExprType::Min:
      case symbolic::details::ExprType::Max:
      case symbolic::details::ExprType::Const:
      case symbolic::details::ExprType::Identity:
      case symbolic::details::ExprType::NonAffine:
        diag::unreachable();
      }
      if (opcode.has_value()) {
        memory::NodeId sid = adjSupergraph.addNode(SymValue{
            .original = s,
        });
        symbolToNodeMap[s] = sid;
        if (lhs.isConstant() && rhs.isConstant()) {
          diag::invalid_state();
        } else if (lhs.isConstant()) {
          assert(rhs.isSymbolic());
          memory::NodeId rid = symbolToNodeMap[rhs.sym()];
          assert(static_cast<bool>(rid));
          adjSupergraph.addEdge(rid, sid, SymOp{*opcode, lhs.constant()},
                                weight);
        } else if (rhs.isConstant()) {
          assert(lhs.isSymbolic());
          memory::NodeId lid = symbolToNodeMap[expr.lhs.sym()];
          assert(lid);
          adjSupergraph.addEdge(lid, sid, SymOp{*opcode, rhs.constant()},
                                weight);
        } else {
          assert(lhs.isSymbolic());
          assert(rhs.isSymbolic());
          memory::NodeId src0 = symbolToNodeMap[lhs.sym()];
          memory::NodeId src1 = symbolToNodeMap[rhs.sym()];
          assert(src0);
          assert(src1);
          adjSupergraph.addEdge(src0, src1, sid, SymOp{*opcode}, weight);
        }
      } else {
        remap[s] = Sym::Const(constant);
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
          break;
        case symbolic::details::ExprType::Div: {
          Sym lhs = resolve(nonaffine.symbols[0]);
          Sym rhs = resolve(nonaffine.symbols[1]);
          memory::NodeId sid = adjSupergraph.addNode(SymValue{
              .original = s,
          });
          symbolToNodeMap[s] = sid;

          if (lhs.isConstant() && rhs.isConstant()) {
            diag::invalid_state();
          } else if (lhs.isConstant()) {
            Sym::value_type constant = lhs.constant();
            SymIROpCode opcode = SymIROpCode::Div_CS;
            weight_type weight = DIV_CS_WEIGHT;
            assert(rhs.isSymbolic());
            memory::NodeId rid = symbolToNodeMap[rhs.sym()];
            assert(rid);
            adjSupergraph.addEdge(rid, sid, SymOp{opcode, constant}, weight);
          } else if (rhs.isConstant()) {
            Sym::value_type constant = rhs.constant();
            SymIROpCode opcode = SymIROpCode::Div_SC;
            weight_type weight = DIV_SC_WEIGHT;
            assert(rhs.constant() != 0);
            assert(lhs.isSymbolic());
            memory::NodeId lid = symbolToNodeMap[lhs.sym()];
            assert(lid);
            adjSupergraph.addEdge(lid, sid, SymOp{opcode, constant}, weight);
          } else {
            SymIROpCode opcode = SymIROpCode::Div_SS;
            weight_type weight = DIV_SS_WEIGHT;
            assert(lhs.isSymbolic());
            assert(rhs.isSymbolic());
            memory::NodeId src0 = symbolToNodeMap[lhs.sym()];
            memory::NodeId src1 = symbolToNodeMap[rhs.sym()];
            adjSupergraph.addEdge(src0, src1, sid, SymOp{opcode}, weight);
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

          if (lhs.isConstant() && rhs.isConstant()) {
            diag::invalid_state();
          } else if (lhs.isConstant()) {
            Sym::value_type constant = lhs.constant();
            SymIROpCode opcode = SymIROpCode::Mod_CS;
            weight_type weight = MOD_CS_WEIGHT;
            assert(rhs.isSymbolic());
            memory::NodeId rid = symbolToNodeMap[rhs.sym()];
            assert(rid);
            adjSupergraph.addEdge(rid, sid, SymOp{opcode, constant}, weight);
          } else if (rhs.isConstant()) {
            Sym::value_type constant = rhs.constant();
            SymIROpCode opcode = SymIROpCode::Mod_SC;
            weight_type weight = MOD_SC_WEIGHT;
            assert(rhs.constant() != 0);
            assert(lhs.isSymbolic());
            memory::NodeId lid = symbolToNodeMap[lhs.sym()];
            assert(lid);
            adjSupergraph.addEdge(lid, sid, SymOp{opcode, constant}, weight);
          } else {
            SymIROpCode opcode = SymIROpCode::Mod_SS;
            weight_type weight = MOD_SS_WEIGHT;
            assert(lhs.isSymbolic());
            assert(rhs.isSymbolic());
            memory::NodeId src0 = symbolToNodeMap[lhs.sym()];
            memory::NodeId src1 = symbolToNodeMap[rhs.sym()];
            assert(src0);
            assert(src1);
            adjSupergraph.addEdge(src0, src1, sid, SymOp{opcode}, weight);
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

            SymIROpCode opcode = SymIROpCode::Mul_SC;
            adjSupergraph.addEdge(src, sid, SymOp{opcode, K}, MUL_SC_WEIGHT);
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

          SymIROpCode opcode = SymIROpCode::Mul_SS;
          for (const auto &perm : perms) {
            memory::vector<memory::NodeId> intermediates(perm.ops.size());
            for (std::size_t o = 0; o < perm.ops.size(); ++o) {
              const auto &op = perm.ops[o];
              memory::NodeId lhs;
              if (op.lhsIntermediate) {
                lhs = intermediates[op.lhs];
                assert(lhs);
              } else {
                lhs = symbolToNodeMap[factors[op.lhs]];
                assert(lhs);
              }
              memory::NodeId rhs;
              if (op.rhsIntermediate) {
                rhs = intermediates[op.rhs];
                assert(rhs);
              } else {
                rhs = symbolToNodeMap[factors[op.rhs]];
                assert(rhs);
              }
              auto key =
                  std::make_pair(std::min(*lhs, *rhs), std::max(*lhs, *rhs));
              if (o == perm.ops.size() - 1) {
                const auto it = mulCache.find(key);
                if (it == mulCache.end()) {
                  adjSupergraph.addEdge(lhs, rhs, factorProd, SymOp{opcode},
                                        MUL_SS_WEIGHT);
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
                adjSupergraph.addEdge(lhs, rhs, dst, SymOp{opcode},
                                      MUL_SS_WEIGHT);
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
            SymIROpCode opcode = SymIROpCode::Mul_SC;
            adjSupergraph.addEdge(factorProd, sid, SymOp{opcode, K},
                                  MUL_SC_WEIGHT);
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
            assert(src);

            SymIROpCode opcode = SymIROpCode::Min_SC;
            adjSupergraph.addEdge(src, sid, SymOp{opcode, K}, MIN_SC_WEIGHT);
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

          SymIROpCode opcode = SymIROpCode::Min_SS;
          for (const auto &perm : perms) {
            memory::vector<memory::NodeId> intermediates(perm.ops.size());
            for (std::size_t o = 0; o < perm.ops.size(); ++o) {
              const auto &op = perm.ops[o];
              memory::NodeId lhs;
              if (op.lhsIntermediate) {
                lhs = intermediates[op.lhs];
                assert(lhs);
              } else {
                lhs = symbolToNodeMap[factors[op.lhs]];
                assert(lhs);
              }
              memory::NodeId rhs;
              if (op.rhsIntermediate) {
                rhs = intermediates[op.rhs];
                assert(rhs);
              } else {
                rhs = symbolToNodeMap[factors[op.rhs]];
                assert(rhs);
              }
              auto key =
                  std::make_pair(std::min(*lhs, *rhs), std::max(*lhs, *rhs));
              if (o == perm.ops.size() - 1) {
                const auto it = minCache.find(key);
                if (it == minCache.end()) {
                  adjSupergraph.addEdge(lhs, rhs, factorProd, SymOp{opcode},
                                        MIN_SS_WEIGHT);
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
                adjSupergraph.addEdge(lhs, rhs, dst, SymOp{opcode},
                                      MIN_SS_WEIGHT);
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
            SymIROpCode opcode = SymIROpCode::Min_SC;
            adjSupergraph.addEdge(factorProd, sid, SymOp{opcode, K},
                                  MIN_SC_WEIGHT);
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
            assert(src);

            SymIROpCode opcode = SymIROpCode::Max_SC;
            adjSupergraph.addEdge(src, sid, SymOp{opcode, K}, MAX_SC_WEIGHT);
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

          SymIROpCode opcode = SymIROpCode::Max_SS;
          for (const auto &perm : perms) {
            memory::vector<memory::NodeId> intermediates(perm.ops.size());
            for (std::size_t o = 0; o < perm.ops.size(); ++o) {
              const auto &op = perm.ops[o];
              memory::NodeId lhs;
              if (op.lhsIntermediate) {
                lhs = intermediates[op.lhs];
                assert(lhs);
              } else {
                lhs = symbolToNodeMap[factors[op.lhs]];
                assert(lhs);
              }
              memory::NodeId rhs;
              if (op.rhsIntermediate) {
                rhs = intermediates[op.rhs];
                assert(rhs);
              } else {
                rhs = symbolToNodeMap[factors[op.rhs]];
                assert(rhs);
              }
              auto key =
                  std::make_pair(std::min(*lhs, *rhs), std::max(*lhs, *rhs));
              if (o == perm.ops.size() - 1) {
                const auto it = maxCache.find(key);
                if (it == maxCache.end()) {
                  adjSupergraph.addEdge(lhs, rhs, factorProd, SymOp{opcode},
                                        MAX_SS_WEIGHT);
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
                adjSupergraph.addEdge(lhs, rhs, dst, SymOp{opcode},
                                      MAX_SS_WEIGHT);
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
            SymIROpCode opcode = SymIROpCode::Max_SC;
            adjSupergraph.addEdge(factorProd, sid, SymOp{opcode, K},
                                  MAX_SC_WEIGHT);
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

  // assert(adjSupergraph.nodeCount() >= m_expressions.size());
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
      results.push_back(nid);
    }
    if (m_expressions[original].expr == ExprType::Identity) {
      vars.push_back(nid);
    }
  }
  ir.varCount = vars.size();

  auto hyperpath =
      *algorithm::shortest_dag_hyperpath(supergraph, vars, results);

  // Reverse map.
  memory::vector<memory::optional<Sym::symbol>> nodeIdToSymbol(
      supergraph.nodeCount());
  for (std::size_t n = 0; n < supergraph.nodeCount(); ++n) {
    memory::NodeId nid{n};
    auto node = supergraph.get(nid);
    nodeIdToSymbol[*nid] = node.original;
  }

  ir.ops.reserve(hyperpath.size());
  memory::vector<std::int64_t> nodeIdToIR(supergraph.nodeCount());
  for (std::int64_t i = 0; i < static_cast<std::int64_t>(vars.size()); ++i) {
    nodeIdToIR[static_cast<std::size_t>(i)] = i;
  }

  for (std::size_t i = 0; i < hyperpath.size(); ++i) {
    memory::EdgeId eid{hyperpath[i]};
    auto op = supergraph.get(eid);
    auto srcs = supergraph.src(eid);
    auto dst = supergraph.dst(eid);
    std::int64_t ird = static_cast<std::int64_t>(i + vars.size());
    nodeIdToIR[*dst] = ird;

    memory::optional<Sym::symbol> original = supergraph.get(dst).original;
    if (original.has_value()) {
      remap[*original] = Sym::Symbol(static_cast<std::uint64_t>(ird));
    }
    SymIROp irop;
    irop.opcode = op.opcode;
    switch (op.opcode) {
    case SymIROpCode::Add_SS:
    case SymIROpCode::Sub_SS:
    case SymIROpCode::Mul_SS:
    case SymIROpCode::Div_SS:
    case SymIROpCode::Mod_SS:
    case SymIROpCode::Min_SS:
    case SymIROpCode::Max_SS:
      irop.lhs = nodeIdToIR[*srcs[0]];
      irop.rhs = nodeIdToIR[*srcs[1]];
      break;
    case SymIROpCode::Add_SC:
    case SymIROpCode::Sub_SC:
    case SymIROpCode::Mul_SC:
    case SymIROpCode::Div_SC:
    case SymIROpCode::Mod_SC:
    case SymIROpCode::Min_SC:
    case SymIROpCode::Max_SC:
      irop.lhs = nodeIdToIR[*srcs[0]];
      irop.rhs = op.constant;
      break;
    case SymIROpCode::Sub_CS:
    case SymIROpCode::Div_CS:
    case SymIROpCode::Mod_CS:
      irop.lhs = op.constant;
      irop.rhs = nodeIdToIR[*srcs[0]];
      break;
    }
    ir.ops.push_back(irop);
  }

  for (std::uint64_t v = 0; v < vars.size(); ++v) {
    auto varNode = supergraph.get(vars[v]);
    assert(varNode.original.has_value());
    Sym::symbol original = varNode.original.value();
    remap[original] = Sym::Symbol(v);
  }

  return std::make_pair(ir, SymRemap(remap));
}

} // namespace denox::compiler
