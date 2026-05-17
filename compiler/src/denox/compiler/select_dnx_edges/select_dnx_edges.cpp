#include "denox/compiler/select_dnx_edges/select_dnx_edges.hpp"
#include "denox/common/SHA256.hpp"
#include "denox/compiler/implement/ComputeDispatch.hpp"
#include "denox/compiler/implement/Supergraph.hpp"
#include "denox/diag/invalid_argument.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/diag/unreachable.hpp"
#include "denox/memory/container/dynamic_bitset.hpp"
#include "denox/memory/container/hashmap.hpp"
#include <algorithm>
#include <cstring>
#include <dnx.h>
#include <stdexcept>
#include <type_traits>

namespace denox::compiler {

SuperGraph select_dnx_edges(memory::span<const std::byte> dnxBuf,
                            SuperGraph &supergraph) {

  const dnx::Model *dnx = dnx::GetModel(dnxBuf.data());
  const uint32_t dnxTensorCount = dnx->tensors()->size();

  memory::vector<Sym> dnxSymbols;

  SymGraph &symGraph = supergraph.symGraph;
  symGraph.debugDump();
  {
    // 1. Determine what SymGraph symbol dnx symbolic variables correspond to!
    const dnx::SymIR *symir = dnx->sym_ir();
    const uint32_t varCount = symir->var_count();
    dnxSymbols.reserve(varCount + symir->ops()->size());
    for (uint32_t v = 0; v < varCount; ++v) {
      // 1. Search dnx for input tensor, which has a
      //    dynamic shape with SymbolicSource(sid)
      const uint32_t inputCount = dnx->inputs()->size();
      for (uint32_t i = 0; i < inputCount; ++i) {
        const dnx::Tensor *dnxTensor =
            dnx->tensors()->Get(dnx->inputs()->Get(i));
        const dnx::TensorInfo *dnxInfo = dnxTensor->info();
        if (dnxInfo == nullptr) {
          diag::invalid_state(
              "dnx without input (interface) tensor infos are invalid!");
        }

        // search for corresponding supergraph input
        auto it = std::ranges::find_if(
            supergraph.inputs, [&](memory::NodeId nid) -> bool {
              const TensorId &tid = supergraph.graph.get(nid);
              const Tensor &tensor = supergraph.tensors[tid.index];
              if (!tensor.info.name.has_value()) {
                return false;
              }
              return dnxInfo->name()->string_view() == *tensor.info.name;
            });
        if (it == supergraph.inputs.end()) {
          diag::invalid_argument("dnx is incompatible with provided model!");
        }
        const TensorInfo &info =
            supergraph.tensors[supergraph.graph.get(*it).index].info;
        if (dnxInfo->height_type() == dnx::ScalarSource_symbolic &&
            dnxInfo->height_as_symbolic()->sid() == v) {
          if (!info.height.has_value() || info.height->isConstant()) {
            diag::invalid_argument("dnx is incompatible with provided model!");
          }
          dnxSymbols.emplace_back(*info.height);
        } else if (dnxInfo->width_type() == dnx::ScalarSource_symbolic &&
                   dnxInfo->width_as_symbolic()->sid() == v) {
          if (!info.width.has_value() || info.width->isConstant()) {
            diag::invalid_argument("dnx is incompatible with provided model!");
          }
          dnxSymbols.emplace_back(*info.width);
        } else if (dnxInfo->channels_type() == dnx::ScalarSource_symbolic &&
                   dnxInfo->channels_as_symbolic()->sid() == v) {
          if (!info.channels.has_value() || info.channels->isConstant()) {
            diag::invalid_argument("dnx is incompatible with provided model!");
          }
          dnxSymbols.emplace_back(*info.channels);
        }
      }
    }
    // 2. Symbolically evaluate SymIR (wtf are we doing xD).
    const uint32_t opCount = symir->ops()->size();
    for (uint32_t o = 0; o < opCount; ++o) {
      const dnx::SymIROp *op = symir->ops()->Get(o);
      dnx::SymIROpCode opcode = op->opcode();

      Sym lhs;
      if (opcode & dnx::SymIROpCode_LHSC) {
        lhs = Sym::Const(op->lhs());
      } else {
        lhs = dnxSymbols[static_cast<size_t>(op->lhs())];
      }
      Sym rhs;
      if (opcode & dnx::SymIROpCode_RHSC) {
        rhs = Sym::Const(op->rhs());
      } else {
        rhs = dnxSymbols[static_cast<size_t>(op->rhs())];
      }
      using ut = std::underlying_type_t<dnx::SymIROpCode>;
      opcode = static_cast<dnx::SymIROpCode>((static_cast<ut>(opcode) & ~(static_cast<ut>(dnx::SymIROpCode_LHSC) |
                           static_cast<ut>(dnx::SymIROpCode_RHSC))));

      Sym &out = dnxSymbols.emplace_back();

      if (opcode == dnx::SymIROpCode_ADD) {
        out = symGraph.add(lhs, rhs);
      } else if (opcode == dnx::SymIROpCode_SUB) {
        out = symGraph.sub(lhs, rhs);
      } else if (opcode == dnx::SymIROpCode_MUL) {
        out = symGraph.mul(lhs, rhs);
      } else if (opcode == dnx::SymIROpCode_DIV) {
        out = symGraph.div(lhs, rhs, false, false);
        // NOTE: Theoretically modproofs, should actually be fine here, because
        // we should theoretically only go through paths, that we have already
        // looked at, thereby modproofs should not blow up memory
        // but something is definitely not working right now.
      } else if (opcode == dnx::SymIROpCode_MOD) {
        out = symGraph.mod(lhs, rhs);
      } else if (opcode == dnx::SymIROpCode_MIN) {
        out = symGraph.min(lhs, rhs);
      } else if (opcode == dnx::SymIROpCode_MAX) {
        out = symGraph.max(lhs, rhs);
      } else {
        diag::invalid_state("Invalid dnx symir");
      }
    }

    // for (Sym s : dnxSymbols) {
    //   fmt::println("DnxSymbol: {} : {}", s, symGraph.to_string(s));
    // }

    // fmt::println("symGraph symbol-count: {}", symGraph.symbolCount());
  }

  memory::dynamic_bitset dnxIsParameter(dnxTensorCount);
  for (uint32_t i = 0; i < dnx->initializers()->size(); ++i) {
    const dnx::TensorInitializer *init = dnx->initializers()->Get(i);
    dnxIsParameter[init->tensor()] = true;
  }

  const uint32_t dnxDispatchCount = dnx->dispatches()->size();
  for (uint32_t d = 0; d < dnxDispatchCount; ++d) {
    const dnx::ComputeDispatch *dnxDispatch = dnx->dispatches()->Get(d);

    // Check if the dispatch has parameters
    // (i.e. bindings to tensors, which are referenced by a initializer)
    bool dnxIsParameterized = false;
    const uint32_t setCount = dnxDispatch->bindings()->size();
    for (uint32_t s = 0; s < setCount; ++s) {
      const dnx::DescriptorSetBinding *setBinding =
          dnxDispatch->bindings()->Get(s);
      const uint32_t bindingCount = setBinding->bindings()->size();
      for (uint32_t b = 0; b < bindingCount; ++b) {
        const dnx::DescriptorBinding *binding = setBinding->bindings()->Get(b);
        uint32_t tid = binding->tensor();
        if (dnxIsParameter[tid]) {
          dnxIsParameterized = true;
        }
      }
    }
    if (!dnxIsParameterized) {
      fmt::println("dispatch without parameters was skipped");
      continue; // dispatches without parameters are uninteressting.
    }

    struct Candidate {
      memory::EdgeId eid;
      uint32_t did; // dispatchID
    };

    memory::small_vector<Candidate, 2> candidates;
    {
      const uint32_t dnxBinaryId = dnxDispatch->binary_id();
      const dnx::ShaderBinary *dnxBinary =
          dnx->shader_binaries()->Get(dnxBinaryId);
      SHA256 dnxSourceHash;
      std::memcpy(dnxSourceHash.h, dnxBinary->source_hash()->hash()->data(),
                  sizeof(uint32_t) * 8);
      for (uint32_t e = 0; e < supergraph.graph.edgeCount(); ++e) {
        const memory::EdgeId eid{e};
        const SuperGraphEdge &edge = supergraph.graph.get(eid);
        for (uint32_t did = 0; did < edge.dispatches.size(); ++did) {
          const ComputeDispatch &dispatch = edge.dispatches[did];
          SHA256 sourceHash = dispatch.glsl.fast_sha256();
          if (sourceHash != dnxSourceHash) {
            continue; // different code
          }

          // fmt::println("CANDIDATE:");
          // fmt::println("dnx-name: {}",
          //              dnxDispatch->info()->name()->string_view());
          // fmt::println("    name: {}", *dispatch.info.name);
          // fmt::println("      op: {}", *dispatch.info.operation);

          // NOTE: This is where it get's really really difficult!

          { // check workgroup count X
            if (dnxDispatch->workgroup_count_x_type() ==
                dnx::ScalarSource_symbolic) {
              Sym dnxSym =
                  dnxSymbols[dnxDispatch->workgroup_count_x_as_symbolic()
                                 ->sid()];
              if (dnxSym != dispatch.workgroupCountX) {
                continue; // different workgroupCountX
              }
            } else {
              // TODO: constant comparison.
            }
          }

          { // check workgroup count Y
            if (dnxDispatch->workgroup_count_y_type() ==
                dnx::ScalarSource_symbolic) {
              Sym dnxSym =
                  dnxSymbols[dnxDispatch->workgroup_count_y_as_symbolic()
                                 ->sid()];
              if (dnxSym != dispatch.workgroupCountY) {
                fmt::println("workgroup-count-y:\ndnx: {} : {}\n     {} : {}",
                             dnxSym, symGraph.to_string(dnxSym),
                             dispatch.workgroupCountY,
                             symGraph.to_string(dispatch.workgroupCountY));
                continue; // different workgroupCountY
              }
            } else {
              // TODO constant comparison
            }
          }
          { // check workgroup count Z
            if (dnxDispatch->workgroup_count_z_type() ==
                dnx::ScalarSource_symbolic) {
              Sym dnxSym =
                  dnxSymbols[dnxDispatch->workgroup_count_z_as_symbolic()
                                 ->sid()];
              if (dnxSym != dispatch.workgroupCountZ) {
                fmt::println("workgroup-count-y:\ndnx: {} : {}\n     {} : {}",
                             dnxSym, symGraph.to_string(dnxSym),
                             dispatch.workgroupCountZ,
                             symGraph.to_string(dispatch.workgroupCountZ));
                continue; // different workgroupCountY
              }
            } else {
              // TODO constant comparison
            }
          }

          // fmt::println("dnx push-constants:");
          // const dnx::PushConstant *pc = dnxDispatch->push_constant();
          // const uint32_t pcCount = pc->fields()->size();
          // for (uint32_t p = 0; p < pcCount; ++p) {
          //   const dnx::PushConstantField *field =
          //       dnxDispatch->push_constant()->fields()->Get(p);
          //   if (field->source_type() == dnx::ScalarSource_symbolic) {
          //     uint32_t sid = field->source_as_symbolic()->sid();
          //     Sym s = dnxSymbols[sid];
          //     fmt::println("pc[{}]: {} -> {}", p, s, symGraph.to_string(s));
          //   }
          // }
          // fmt::println("push constants:");
          // for (const auto &pc : dispatch.pushConstants) {
          //   Sym s = pc.sym();
          //   fmt::println("pc[]: {} -> {}", s, symGraph.to_string(s));
          // }

          // { // check workgroup count Z
          //   if (dnxDispatch->workgroup_count_z_type() ==
          //       dnx::ScalarSource_symbolic) {
          //     Sym dnxSym =
          //         dnxSymbols[dnxDispatch->workgroup_count_z_as_symbolic()
          //                        ->sid()];
          //     if (dnxSym != dispatch.workgroupCountZ) {
          //       continue; // different workgroupCountX
          //     }
          //   }
          // }

          // dnxDispatch->workgroup_count_x_as_symbolic();
          // dispatch.workgroupCountX;

          candidates.emplace_back(eid, did);
        }
      }
    }
    fmt::println("candidates: {}", candidates.size());
  }
  throw std::runtime_error("work-in-progress");
}

} // namespace denox::compiler
