#include "compiler/impl/impl.hpp"
#include "algorithm/pattern_matching/ConstGraphMatch.hpp"
#include "algorithm/pattern_matching/match.hpp"
#include "algorithm/shortest_dag_hyperpath.hpp"
#include "diag/logging.hpp"
#include "heuristic/IHeuristic.hpp"
#include "heuristic/MemoryHeuristic.hpp"
#include "memory/container/vector.hpp"
#include "memory/coroutines/generator.hpp"
#include "memory/hypergraph/AdjGraph.hpp"
#include "memory/hypergraph/ConstGraph.hpp"
#include "shaders/GlslCompiler.hpp"
#include "shaders/IShader.hpp"
#include "shaders/activation/BasicActivationShader.hpp"
#include "shaders/conv/DirectConvShader.hpp"
#include "shaders/copy/CopyTransformShader.hpp"
#include "shaders/pad/MemoryPadShader.hpp"
#include "shaders/pool/BasicPoolShader.hpp"
#include "shaders/slice/MemorySliceShader.hpp"
#include "shaders/upsample/BasicUpsampleShader.hpp"

namespace denox::compiler {

struct ComputeOpImpl {
  const IShader *shader;
  unsigned int pattern;
  algorithm::ConstGraphMatch<TensorInstance, ComputeOp> match;
};

using SuperGraph = memory::AdjGraph<TensorInstance, ComputeOpImpl, float>;

ImplModel implement(const OpModel &model, const SymGraph &symGraphRef) {
  SymGraph symGraph = symGraphRef;
  const auto &opGraph = model.graph;
  SuperGraph supergraph{};

  // NOTE: Supergraph has compatible nodeIds with OpModel!
  for (std::uint64_t n = 0; n < opGraph.nodeCount(); ++n) {
    memory::NodeId nid{n};
    supergraph.addNode(opGraph.get(nid));
  }

  GlslCompiler glslCompiler;

  shaders::DirectConvShader directConv{&glslCompiler};
  shaders::BasicPoolShader basicPool{&glslCompiler};
  shaders::BasicUpsampleShader basicUpsample{&glslCompiler};
  shaders::MemoryPadShader memoryPad{&glslCompiler};
  shaders::MemorySliceShader memorySlice{&glslCompiler};
  shaders::BasicActivationShader basicActivation{&glslCompiler};
  shaders::CopyTransformShader copyTransform{&glslCompiler};

  const IShader *shaders[]{
      &directConv,      //
      &basicPool,       //
      &basicUpsample,   //
      &memoryPad,       //
      &memorySlice,     //
      &basicActivation, //
      &copyTransform,
  };

  MemoryHeuristic memoryHeuristic{shaders, &opGraph, symGraphRef, model.input};

  IHeuristic *heuristic = &memoryHeuristic;

  std::size_t nodeCount = opGraph.nodeCount();

  constexpr std::size_t sn = sizeof(shaders) / sizeof(IShader *);
  for (std::size_t s = 0; s < sn; ++s) {

    const IShader *shader = shaders[s];
    const ShaderCapabilities &caps = shader->capabilities();
    const unsigned int pn = static_cast<unsigned int>(caps.patterns.size());


    algorithm::match_all<TensorInstance, ComputeOp>(caps.patterns[0].pattern, opGraph);
    

    denox::memory::generator<algorithm::ConstGraphMatch<int, int>> x;

    for (unsigned int p = 0; p < pn; ++p) {
      memory::dynamic_bitset edgeExits(nodeCount * nodeCount * nodeCount,
                                       false);
    for (const auto &m :
         algorithm::match_all(caps.patterns[p].pattern, opGraph)) {
          memory::small_vector<memory::NodeId, 2> inputs;
          memory::small_vector<const TensorInstance *, 2> ins;
          for (std::size_t i = 0; i < caps.patterns[p].inputs.size(); ++i) {
            memory::NodeId in = m[caps.patterns[p].inputs[i]];

            inputs.push_back(in);
            ins.push_back(&opGraph.get(in));
          }
          memory::NodeId out = m[caps.patterns[p].output];
          std::uint64_t edgeId =
              static_cast<std::uint64_t>(inputs[0]) * nodeCount +
              static_cast<std::uint64_t>(out);
          if (inputs.size() == 2) {
            edgeId += inputs[1] * nodeCount * nodeCount;
          }
          if (edgeExits[edgeId]) {
            continue;
          }
          auto pattern = shader->acceptMatch(opGraph, p, m);
          if (!pattern.has_value()) {
            continue;
          }

          edgeExits[edgeId] = true;

          const float w =
              heuristic->eval(ins, opGraph.get(out), *pattern, m, shader);

          supergraph.addEdge(inputs, out,
                             ComputeOpImpl{
                                 .shader = shader,
                                 .pattern = *pattern,
                                 .match = m,
                             },
                             w);
    }
    }
  }
  memory::ConstGraph<TensorInstance, ComputeOpImpl, float> constSupergraph{
      supergraph};

  memory::span<const memory::NodeId> starts{&model.input, 1};
  memory::span<const memory::NodeId> ends{&model.output, 1};
  auto opthyperpath =
      algorithm::shortest_dag_hyperpath<TensorInstance, ComputeOpImpl, float>(
          constSupergraph, starts, ends);
  if (!opthyperpath.has_value()) {
    DENOX_ERROR("Failed to implement model.");
    std::terminate(); // <- TODO proper error handling please
  }
  const memory::vector<memory::EdgeId> &hyperpath = *opthyperpath;

  { // Printing the hyperedge.
    fmt::println("\x1B[32m\x1B[1m{:=^100}\x1B[0m",
                 "Selected=Execution=Schedule");
    float totalWeight = 0;
    for (std::size_t op = 0; op < hyperpath.size(); ++op) {
      memory::EdgeId oid{(hyperpath)[op]};
      const auto &o = constSupergraph.get(oid);
      const auto &srcs = constSupergraph.src(oid);
      bool first = true;
      std::string inStr;
      for (const memory::NodeId &srcId : srcs) {
        const TensorInstance &src = constSupergraph.get(srcId);
        if (!first) {
          inStr += ",";
        }
        first = false;
        inStr += fmt::format("{}[{}]", src.layout.to_string(), src.channels);
      }

      const ComputeOpImpl &impl = constSupergraph.get(oid);
      memory::NodeId dstId = constSupergraph.dst(oid);
      const TensorInstance &dst = constSupergraph.get(dstId);
      totalWeight += constSupergraph.weight(oid);
      fmt::println("{:>22} \x1B[34m{:-^40}>\x1B[0m {:<22} : {}", inStr,
                   impl.shader->name(o.pattern),
                   fmt::format("{}[{}]", dst.layout.to_string(),
                   dst.channels),
                   heuristic->weight_to_string(constSupergraph.weight(oid)));
    }
    fmt::println("\x1B[31m{:>89} {}\x1B[0m",
                 "Total Cost:", heuristic->weight_to_string(totalWeight));
  }

  ImplModel implModel;
  Impl impl{&implModel};

  // 1. Collect all intermediate tensors.
  TensorId input;
  TensorId output;
  for (const auto &opId : hyperpath) {
    const ComputeOpImpl &op = constSupergraph.get(opId);
    memory::span<const memory::NodeId> srcs = constSupergraph.src(opId);
    for (const memory::NodeId src : srcs) {
      TensorId tensorId =
          impl.createTensor(symGraph, constSupergraph.get(src), src);
      if (src == model.input) {
        input = tensorId;
      }
    }
    memory::NodeId dst = constSupergraph.dst(opId);
    TensorId dstTensorId =
        impl.createTensor(symGraph, constSupergraph.get(dst), dst);
    if (dst == model.output) {
      output = dstTensorId;
    }

    op.shader->implement(impl, opGraph, op.pattern, op.match);
  }
  assert(input.index != TensorId::nullindex);
  assert(output.index != TensorId::nullindex);

  fmt::println("\n\x1B[32m\x1B[1m{:=^100}\x1B[0m", "Implemented=Schedule");

  fmt::println("=> \x1B[35m\x1B[1m{}\x1B[0m: {}", "Input", input.index);
  memory::hash_map<Sym::symbol, memory::string> symbolNames;
  sym inputWidthSym = constSupergraph.get(model.input).extent.x;
  if (inputWidthSym.isSymbolic()) {
    symbolNames.emplace(inputWidthSym.symbol(), "Input.Width");
  }
  sym inputHeightSym = constSupergraph.get(model.input).extent.y;
  if (inputHeightSym.isSymbolic()) {
    symbolNames.emplace(inputHeightSym.symbol(), "Input.Height");
  }

  for (std::size_t d = 0; d < implModel.dispatches.size(); ++d) {
    const auto &dispatch = implModel.dispatches[d];
    memory::string dispatchName;
    memory::string sourcePath;
    if (dispatch.meta != nullptr && dispatch.meta->name.has_value()) {
      dispatchName = fmt::format("{}", *dispatch.meta->name);
    } else {
      dispatchName = fmt::format("unnamed-dispatch");
    }
    if (dispatch.meta != nullptr && dispatch.meta->sourcePath.has_value()) {
      sourcePath = dispatch.meta->sourcePath.value().str();
    } else {
      sourcePath = "<unknown>";
    }
    fmt::println("\x1B[34m\x1B[1m{}\x1B[0m: (\x1B[4m{}\x1B[0m)",
    dispatchName,
                 sourcePath);
    fmt::print("\u2022 TensorBindings: [");
    bool first = true;
    for (const auto &binding : dispatch.bindings) {
      if (!first) {
        fmt::print(", ");
      }
      first = false;
      fmt::print("{}", binding.tensorId.index);
    }
    fmt::println("]");
    fmt::println("\u2022 PushConstants: ");
    for (std::size_t p = 0; p < dispatch.pushConstants.size(); ++p) {
      const auto &pushConstant = dispatch.pushConstants[p];
      fmt::println("  - {}", pushConstant.to_string(symGraph, symbolNames));
    }
  }
  fmt::println("=> \x1B[35m\x1B[1m{}\x1B[0m: {}", "Output", output.index);

  fmt::println("\n\x1B[31mSummary:\x1B[0m");

  fmt::println("\u2022 {:<20} : {}", "Number-Of-Dispatches",
               implModel.dispatches.size());

  std::size_t parameterByteSize = 0;
  for (std::size_t p = 0; p < implModel.parameters.size(); ++p) {
    parameterByteSize += implModel.parameters[p].data.size();
  }
  if (parameterByteSize > 1000000) {
    fmt::println("\u2022 {:<20} : {:.1f}MB", "Parameter-ByteSize",
                 static_cast<float>(parameterByteSize) / 1000000.0f);
  } else if (parameterByteSize > 1000) {
    fmt::println("\u2022 Parameter-ByteSize : {:.1f}KB",
                 static_cast<float>(parameterByteSize) / 1000.0f);
  } else {
    fmt::println("\u2022 {:<20} : {}B", "Parameter-ByteSize",
                 parameterByteSize);
  }

  return implModel;
}

} // namespace denox::compiler
