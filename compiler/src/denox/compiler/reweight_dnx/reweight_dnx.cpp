#include "denox/compiler/reweight_dnx/reweight_dnx.hpp"
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
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace denox::compiler {

static uint64_t parse_literal(const dnx::ScalarLiteral *literal) {
  switch (literal->dtype()) {
  case dnx::ScalarType_I16: {
    int16_t v;
    std::memcpy(&v, literal->bytes()->data(), sizeof(int16_t));
    return static_cast<uint64_t>(v);
  }
  case dnx::ScalarType_U16: {
    uint16_t v;
    std::memcpy(&v, literal->bytes()->data(), sizeof(uint16_t));
    return static_cast<uint64_t>(v);
  }
  case dnx::ScalarType_I32: {
    int32_t v;
    std::memcpy(&v, literal->bytes()->data(), sizeof(int32_t));
    return static_cast<uint64_t>(v);
  }
  case dnx::ScalarType_U32: {
    uint32_t v;
    std::memcpy(&v, literal->bytes()->data(), sizeof(uint32_t));
    return static_cast<uint64_t>(v);
  }
  case dnx::ScalarType_I64: {
    int64_t v;
    std::memcpy(&v, literal->bytes()->data(), sizeof(int64_t));
    return static_cast<uint64_t>(v);
  }
  case dnx::ScalarType_U64: {
    uint64_t v;
    std::memcpy(&v, literal->bytes()->data(), sizeof(uint64_t));
    return static_cast<uint64_t>(v);
  }
  case dnx::ScalarType_F16:
  case dnx::ScalarType_F32:
  case dnx::ScalarType_F64:
    return std::numeric_limits<uint64_t>::max();
  default:
    diag::unreachable();
  }
}

static memory::optional<memory::Dtype>
parse_dtype(const dnx::ScalarType dtype) {
  switch (dtype) {
  case dnx::ScalarType_I16:
  case dnx::ScalarType_U16:
    return memory::nullopt;
  case dnx::ScalarType_I32:
    return memory::Dtype::I32;
  case dnx::ScalarType_U32:
    return memory::Dtype::U32;
  case dnx::ScalarType_I64:
    return memory::Dtype::I64;
  case dnx::ScalarType_U64:
    return memory::Dtype::U64;
  case dnx::ScalarType_F16:
    return memory::Dtype::F16;
  case dnx::ScalarType_F32:
    return memory::Dtype::F32;
  case dnx::ScalarType_F64:
    return memory::Dtype::F64;
  default:
    diag::unreachable();
  }
}

void reweight_dnx(memory::span<std::byte> dnxBuf, SuperGraph &supergraph) {

  dnx::Model *dnx = dnx::GetMutableModel(dnxBuf.data());
  const uint32_t dnxTensorCount = dnx->tensors()->size();

  memory::vector<Sym> dnxSymbols;

  SymGraph &symGraph = supergraph.symGraph;
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
      opcode = static_cast<dnx::SymIROpCode>((
          static_cast<ut>(opcode) & ~(static_cast<ut>(dnx::SymIROpCode_LHSC) |
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
  }

  memory::dynamic_bitset dnxIsParameter(dnxTensorCount, false);
  for (uint32_t i = 0; i < dnx->initializers()->size(); ++i) {
    const dnx::TensorInitializer *init = dnx->initializers()->Get(i);
    dnxIsParameter[init->tensor()] = true;
  }

  memory::dynamic_bitset isLive(supergraph.tensors.size(), false);
  for (const memory::NodeId input : supergraph.inputs) {
    TensorId tid = supergraph.graph.get(input);
    isLive[tid.index] = true;
  }
  for (uint64_t e = 0; e < supergraph.graph.edgeCount(); ++e) {
    const memory::EdgeId eid{e};
    for (const auto &param : supergraph.graph.get(eid).parameters) {
      isLive[param.tensorId.index] = true;
    }
  }

  struct Candidate {
    memory::EdgeId eid;
    uint32_t did; // dispatchID
  };
  struct CandidateHash {
    size_t operator()(const Candidate& candidate) const {
      return 0;
    }
  };
  struct CandidateComp {
    bool operator()(const Candidate& lhs, const Candidate& rhs) const {
      return lhs.eid == rhs.eid && lhs.did == rhs.did;
    }
  };
  memory::hash_set<Candidate, CandidateHash, CandidateComp> used;

      const uint32_t dnxDispatchCount = dnx->dispatches()->size();
  for (uint32_t d = 0; d < dnxDispatchCount; ++d) {
    const dnx::ComputeDispatch *dnxDispatch = dnx->dispatches()->Get(d);

    // Check if the dispatch has parameters
    // (i.e. bindings to tensors, which are referenced by a initializer)
    memory::small_vector<uint32_t, 4> dnxParameterTensorIds;
    const uint32_t setCount = dnxDispatch->bindings()->size();
    for (uint32_t s = 0; s < setCount; ++s) {
      const dnx::DescriptorSetBinding *setBinding =
          dnxDispatch->bindings()->Get(s);
      const uint32_t bindingCount = setBinding->bindings()->size();
      for (uint32_t b = 0; b < bindingCount; ++b) {
        const dnx::DescriptorBinding *binding = setBinding->bindings()->Get(b);
        uint32_t tid = binding->tensor();
        if (dnxIsParameter[tid]) {
          dnxParameterTensorIds.push_back(tid);
        }
      }
    }
    if (dnxParameterTensorIds.empty()) {
      continue; // dispatches without parameters are uninteressting.
    }

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
          Candidate candidate{
            .eid = eid,
            .did = did,
          };
          if (used.contains(candidate)) {
            continue;;
          }

          { // check workgroup count X
            Sym dnxSym;
            if (dnxDispatch->workgroup_count_x_type() ==
                dnx::ScalarSource_symbolic) {
              dnxSym = dnxSymbols[dnxDispatch->workgroup_count_x_as_symbolic()
                                      ->sid()];
            } else {
              dnxSym = Sym::Const(
                  parse_literal(dnxDispatch->workgroup_count_x_as_literal()));
            }
            if (dnxSym != dispatch.workgroupCountX) {
              continue; // different workgroupCountX
            }
          }

          { // check workgroup count Y
            Sym dnxSym;
            if (dnxDispatch->workgroup_count_y_type() ==
                dnx::ScalarSource_symbolic) {
              dnxSym = dnxSymbols[dnxDispatch->workgroup_count_y_as_symbolic()
                                      ->sid()];
            } else {
              dnxSym = Sym::Const(
                  parse_literal(dnxDispatch->workgroup_count_y_as_literal()));
            }
            if (dnxSym != dispatch.workgroupCountY) {
              continue; // different workgroupCountY
            }
          }
          { // check workgroup count Z
            Sym dnxSym;
            if (dnxDispatch->workgroup_count_z_type() ==
                dnx::ScalarSource_symbolic) {
              dnxSym = dnxSymbols[dnxDispatch->workgroup_count_z_as_symbolic()
                                      ->sid()];
            } else {
              dnxSym = Sym::Const(
                  parse_literal(dnxDispatch->workgroup_count_z_as_literal()));
            }
            if (dnxSym != dispatch.workgroupCountZ) {
              continue; // different workgroupCountY
            }
          }

          const dnx::PushConstant *pc = dnxDispatch->push_constant();
          const uint32_t pcCount = pc->fields()->size();
          if (pcCount != dispatch.pushConstants.size()) {
            continue;
          }
          bool match = true;
          for (uint32_t p = 0; p < pcCount; ++p) {
            const dnx::PushConstantField *dnxField =
                dnxDispatch->push_constant()->fields()->Get(p);
            const auto &field = dispatch.pushConstants[p];

            auto dnxType = parse_dtype(dnxField->dtype());
            if (!dnxType.has_value()) {
              match = false;
              break;
            }

            if (field.type() != dnxType.value()) {
              match = false;
              break;
            }

            Sym dnxValue;
            if (dnxField->source_type() == dnx::ScalarSource_symbolic) {
              dnxValue = dnxSymbols[dnxField->source_as_symbolic()->sid()];
            } else {
              dnxValue =
                  Sym::Const(parse_literal(dnxField->source_as_literal()));
            }
            if (field.sym() != dnxValue) {
              match = false;
              break;
            }
          }
          if (!match) {
            continue;
          }

          bool unalive = false;
          for (const auto &binding : dispatch.bindings) {
            if ((binding.accessFlag == Access::ReadOnly ||
                 binding.accessFlag == Access::ReadWrite) &&
                !isLive[binding.tensorId.index]) {
              unalive = true;
              break;
            }
          }
          if (unalive) {
            continue;
          }

          candidates.push_back(candidate);
        }
      }
    }

    if (candidates.size() == 0) {
      diag::invalid_argument("Failed to reweight! Reference ONNX model, does "
                             "not seem to be compatible with DNX artefact!");
    } else if (candidates.size() > 1) {
      diag::invalid_state(
          "Failed to map dispatch to reconstructed supergraph, "
          "selection is ambigious!\nTHIS IS A BUG! If you have "
          "the time please create a github issue, we are happy to fix it.");
    }
    const SuperGraphEdge &edge = supergraph.graph.get(candidates.front().eid);
    const ComputeDispatch &dispatch = edge.dispatches[candidates.front().did];

    // make outputs live
    for (const auto &binding : dispatch.bindings) {
      isLive[binding.tensorId.index] = true;
    }
    used.insert(candidates.front());

    for (const uint32_t dnxParamTensorId : dnxParameterTensorIds) {
      uint32_t set = std::numeric_limits<uint32_t>::max();
      uint32_t binding = std::numeric_limits<uint32_t>::max();

      // 1. Determine glsl (set,binding) of dnx parameter!
      {
        bool found = false;
        for (uint32_t s = 0; s < dnxDispatch->bindings()->size() && !found;
             ++s) {
          const dnx::DescriptorSetBinding *dnxSet =
              dnxDispatch->bindings()->Get(s);
          for (uint32_t b = 0; b < dnxSet->bindings()->size(); ++b) {
            const dnx::DescriptorBinding *dnxBinding =
                dnxSet->bindings()->Get(b);
            if (dnxBinding->tensor() == dnxParamTensorId) {
              set = dnxSet->set();
              binding = dnxBinding->binding();
              found = true;
              break;
            }
          }
        }
        if (!found) {
          diag::invalid_state();
        }
        assert(found);
      }
      // 2. Find corresponding tensor binding in dispatch!
      TensorId tensorId{};
      {
        bool found = false;
        for (const auto &b : dispatch.bindings) {
          if (b.binding == binding && b.set == set) {
            tensorId = b.tensorId;
            found = true;
            break;
          }
        }
        if (!found) {
          diag::invalid_state();
        }
      }

      // 3. Get new weight array.
      memory::vector<std::byte> newWeight;
      {
        bool found = false;
        for (const auto &param : edge.parameters) {
          if (param.tensorId.index == tensorId.index) {
            newWeight = param.lazyValue();
            found = true;
            break;
          }
        }
        if (!found) {
          diag::invalid_state();
        }
      }

      // 4. Determine where in the dnx artifact the
      //    corresponding weights where stored.
      {
        bool found = false;
        for (uint32_t i = 0; i < dnx->initializers()->size(); ++i) {
          dnx::TensorInitializer *init =
              dnx->initializers()->GetMutableObject(i);
          if (init->tensor() == dnxParamTensorId) {
            if (init->data()->size() != newWeight.size()) {
              diag::invalid_state();
            }
            uint8_t *oldWeight = init->mutable_data()->data();
            std::memcpy(oldWeight, newWeight.data(), newWeight.size());

            found = true;
            break;
          }
        }
        if (!found) {
          diag::invalid_state();
        }
      }
    }
  }
}

} // namespace denox::compiler
