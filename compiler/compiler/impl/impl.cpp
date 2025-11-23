#include "compiler/impl/impl.hpp"
#include "algorithm/align_up.hpp"
#include "algorithm/hash_combine.hpp"
#include "algorithm/pattern_matching/ConstGraphMatch.hpp"
#include "algorithm/pattern_matching/match.hpp"
#include "algorithm/shortest_dag_hyperpath.hpp"
#include "compiler/impl/ComputeOpImpl.hpp"
#include "compiler/ir/populate/ImplDb.hpp"
#include "diag/failed_to_realize.hpp"
#include "diag/invalid_state.hpp"
#include "diag/not_implemented.hpp"
#include "heuristic/IHeuristic.hpp"
#include "heuristic/MemoryHeuristic.hpp"
#include "memory/container/hashmap.hpp"
#include "memory/container/vector.hpp"
#include "memory/hypergraph/AdjGraph.hpp"
#include "memory/hypergraph/ConstGraph.hpp"
#include "shaders/IShader.hpp"
#include "shaders/compiler/GlslCompiler.hpp"
#include "shaders/shaders.hpp"
#include "symbolic/SymGraphEval.hpp"
#include <limits>
#include <type_traits>
#include <unordered_set>

namespace denox::compiler {

static constexpr size_t DEFAULT_INPUT_WIDTH = 1920;
static constexpr size_t DEFAULT_INPUT_HEIGHT = 1080;

using ComputeOpImpl = impl::details::ComputeOpImpl;

using SuperGraph = memory::AdjGraph<TensorInstance, ComputeOpImpl, float>;

ImplModel implement(const OpModel &model, const SymGraph &symGraphRef,
                    const Options &options) {
  const auto &opGraph = model.graph;
  SuperGraph supergraph{};

  // NOTE: Supergraph has compatible nodeIds with OpModel!
  for (std::uint64_t n = 0; n < opGraph.nodeCount(); ++n) {
    memory::NodeId nid{n};
    supergraph.addNode(opGraph.get(nid));
  }

  GlslCompiler glslCompiler(options);

  const auto shaders = shaders::get_all_shaders(&glslCompiler, options);

  MemoryHeuristic memoryHeuristic{shaders, &opGraph, symGraphRef, model.input};

  IHeuristic *heuristic = &memoryHeuristic;

  std::size_t nodeCount = opGraph.nodeCount();

  std::vector<SymSpec> specs;
  auto in = model.graph.get(model.input);
  if (in.extent.x.isSymbolic()) {
    specs.push_back(SymSpec{
        .symbol = in.extent.x.symbol(),
        .value = DEFAULT_INPUT_WIDTH,
    });
  }
  if (in.extent.y.isSymbolic()) {
    specs.push_back(SymSpec{
        .symbol = in.extent.y.symbol(),
        .value = DEFAULT_INPUT_HEIGHT,
    });
  }

  SymGraphEval symEval0 = symGraphRef.eval(specs);

  std::size_t sn = shaders.size();
  for (std::size_t s = 0; s < sn; ++s) {

    const IShader *shader = shaders[s].get();
    const ShaderCapabilities &caps = shader->capabilities();
    const unsigned int pn = static_cast<unsigned int>(caps.patterns.size());
    for (unsigned int p = 0; p < pn; ++p) {

      std::unordered_set<uint64_t> edgeExists;

      std::uint64_t hash = 0xBBD224F354ADF788;

      for (const auto &m :
           algorithm::match_all(caps.patterns[p].pattern, opGraph)) {
        memory::small_vector<memory::NodeId, 2> inputs;
        memory::small_vector<const TensorInstance *, 2> ins;
        for (std::size_t i = 0; i < caps.patterns[p].inputs.size(); ++i) {
          memory::NodeId in = m[caps.patterns[p].inputs[i]];
          const auto &input = opGraph.get(in);
          inputs.push_back(in);
          ins.push_back(&input);

          hash = algorithm::hash_combine(hash, input.channels);
          hash = algorithm::hash_combine(
              hash, static_cast<uint64_t>(*symEval0[input.extent.x.asSym()]));
          hash = algorithm::hash_combine(
              hash, static_cast<uint64_t>(*symEval0[input.extent.y.asSym()]));

          hash = algorithm::hash_combine(
              hash, static_cast<uint64_t>(input.layout.kind()));
          hash = algorithm::hash_combine(
              hash, static_cast<uint64_t>(input.type.kind()));
        }
        memory::NodeId out = m[caps.patterns[p].output];
        const auto &output = opGraph.get(out);
        std::uint64_t edgeId =
            static_cast<std::uint64_t>(inputs[0]) * nodeCount +
            static_cast<std::uint64_t>(out);
        if (inputs.size() == 2) {
          edgeId += *inputs[1] * nodeCount * nodeCount;
        }
        if (edgeExists.contains(edgeId)) {
          continue;
        }

        auto configs = shader->acceptMatch(opGraph, p, m);
        if (configs.empty()) {
          continue;
        }

        {
          hash = algorithm::hash_combine(hash, output.channels);
          hash = algorithm::hash_combine(
              hash, static_cast<uint64_t>(*symEval0[output.extent.x.asSym()]));
          hash = algorithm::hash_combine(
              hash, static_cast<uint64_t>(*symEval0[output.extent.y.asSym()]));

          hash = algorithm::hash_combine(
              hash, static_cast<uint64_t>(output.layout.kind()));
          hash = algorithm::hash_combine(
              hash, static_cast<uint64_t>(output.type.kind()));
        }

        edgeExists.insert(edgeId);

        for (const unsigned int config : configs) {
          const float w = heuristic->eval(ins, opGraph.get(out), p, config,
                                          hash, m, shader);

          supergraph.addEdge(inputs, out,
                             ComputeOpImpl{
                                 .shader = shader,
                                 .pattern = p,
                                 .config = config,
                                 .match = m,
                             },
                             w);
        }
      }
    }
  }
  fmt::println("edge-count: {}", supergraph.edgeCount());
  memory::ConstGraph<TensorInstance, ComputeOpImpl, float> constSupergraph{
      supergraph};

  memory::span<const memory::NodeId> starts{&model.input, 1};
  memory::span<const memory::NodeId> ends{&model.output, 1};
  auto opthyperpath =
      algorithm::shortest_dag_hyperpath<TensorInstance, ComputeOpImpl, float>(
          constSupergraph, starts, ends);
  if (!opthyperpath.has_value()) {
    compiler::diag::failed_to_realize(model, constSupergraph);
  }
  const memory::vector<memory::EdgeId> &hyperpath = *opthyperpath;

  if (options.verbose) { // Printing the hyperedge.
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
                   impl.shader->name(o.pattern, o.config),
                   fmt::format("{}[{}]", dst.layout.to_string(), dst.channels),
                   heuristic->weight_to_string(constSupergraph.weight(oid)));
    }
    fmt::println("\x1B[31m{:>89} {}\x1B[0m",
                 "Total Cost:", heuristic->weight_to_string(totalWeight));
    fmt::println("");
  }

  ImplModel implModel;
  implModel.symGraph = symGraphRef;
  SymGraph &symGraph = implModel.symGraph;
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
    op.shader->implement(impl, opGraph, op.pattern, op.config, op.match,
                         symGraph);
  }
  assert(input.index != TensorId::nullindex);
  assert(output.index != TensorId::nullindex);
  {
    auto in = model.graph.get(model.input);
    sym_vec2 extent = in.extent;
    implModel.inputs.emplace_back(in.channels, extent, input, in.layout,
                                  in.type);
  }
  {
    auto out = model.graph.get(model.output);
    sym_vec2 extent = out.extent;
    implModel.outputs.emplace_back(out.channels, extent, output, out.layout,
                                   out.type);
  }

  if (!options.skipSpirvCompile) {
    impl.compileAll(!options.quite);
  }

  if (options.verbose) {
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
      fmt::println("\x1B[34m\x1B[1m{}\x1B[0m: (\x1B[4m{}\x1B[0m)", dispatchName,
                   sourcePath);
      fmt::print("\u2022 Binary-Size: {}B\n",
                 implModel.shaderBinaries[dispatch.binaryId].spv.size() *
                     sizeof(std::uint32_t));
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
  }

  return implModel;
}

ImplDb implement_all(const OpModel &model, const SymGraph &symGraphRef,
                     const Options &options) {
  const auto &opGraph = model.graph;

  SuperGraph supergraph{};

  // NOTE: Supergraph has compatible nodeIds with OpModel!
  for (std::uint64_t n = 0; n < opGraph.nodeCount(); ++n) {
    memory::NodeId nid{n};
    supergraph.addNode(opGraph.get(nid));
  }

  GlslCompiler glslCompiler(options);

  const auto shaders = shaders::get_all_shaders(&glslCompiler, options);

  ImplModel implModel;
  implModel.symGraph = symGraphRef;
  SymGraph &symGraph = implModel.symGraph;
  Impl impl{&implModel};

  ImplDb db;

  std::size_t nodeCount = opGraph.nodeCount();
  std::size_t sn = shaders.size();

  // specialize symbolic graph.

  std::vector<SymSpec> specs;
  auto in = model.graph.get(model.input);
  if (in.extent.x.isSymbolic()) {
    specs.push_back(SymSpec{
        .symbol = in.extent.x.symbol(),
        .value = DEFAULT_INPUT_WIDTH,
    });
  }
  if (in.extent.y.isSymbolic()) {
    specs.push_back(SymSpec{
        .symbol = in.extent.y.symbol(),
        .value = DEFAULT_INPUT_HEIGHT,
    });
  }

  SymGraphEval symEval0 = symGraphRef.eval(specs);

  for (std::size_t s = 0; s < sn; ++s) {

    const IShader *shader = shaders[s].get();
    const ShaderCapabilities &caps = shader->capabilities();
    const unsigned int pn = static_cast<unsigned int>(caps.patterns.size());
    for (unsigned int p = 0; p < pn; ++p) {

      std::unordered_set<uint64_t> edgeExists;

      for (const auto &m :
           algorithm::match_all(caps.patterns[p].pattern, opGraph)) {
        memory::small_vector<memory::NodeId, 2> inputs;
        memory::small_vector<TensorId, 2> inputTensors;

        uint64_t hash = 0xBBD224F354ADF788;

        hash = algorithm::hash_combine(hash, caps.patterns[p].inputs.size());

        for (std::size_t i = 0; i < caps.patterns[p].inputs.size(); ++i) {
          memory::NodeId in = m[caps.patterns[p].inputs[i]];
          auto input = opGraph.get(in);
          impl.createTensor(symGraph, input, in);

          hash = algorithm::hash_combine(hash, input.channels);
          hash = algorithm::hash_combine(
              hash, static_cast<uint64_t>(*symEval0[input.extent.x.asSym()]));
          hash = algorithm::hash_combine(
              hash, static_cast<uint64_t>(*symEval0[input.extent.y.asSym()]));

          hash = algorithm::hash_combine(
              hash, static_cast<uint64_t>(input.layout.kind()));
          hash = algorithm::hash_combine(
              hash, static_cast<uint64_t>(input.type.kind()));
        }
        memory::NodeId out = m[caps.patterns[p].output];
        std::uint64_t edgeId =
            static_cast<std::uint64_t>(inputs[0]) * nodeCount +
            static_cast<std::uint64_t>(out);
        if (inputs.size() == 2) {
          edgeId += *inputs[1] * nodeCount * nodeCount;
        }
        if (edgeExists.contains(edgeId)) {
          continue;
        }
        auto output = opGraph.get(out);
        impl.createTensor(symGraph, output, out);
        {
          hash = algorithm::hash_combine(hash, output.channels);
          hash = algorithm::hash_combine(
              hash, static_cast<uint64_t>(*symEval0[output.extent.x.asSym()]));
          hash = algorithm::hash_combine(
              hash, static_cast<uint64_t>(*symEval0[output.extent.y.asSym()]));

          hash = algorithm::hash_combine(
              hash, static_cast<uint64_t>(output.layout.kind()));
          hash = algorithm::hash_combine(
              hash, static_cast<uint64_t>(output.type.kind()));
        }

        auto configs = shader->acceptMatch(opGraph, p, m);
        if (configs.empty()) {
          continue;
        }

        edgeExists.insert(edgeId);

        for (const unsigned int config : configs) {
          size_t begin = implModel.dispatches.size();
          shader->implement(impl, opGraph, p, config, m, symGraph);
          size_t end = implModel.dispatches.size();
          DbOp op;
          op.shaderName = shader->name(p, config);
          op.pattern = p;
          op.config = config;
          op.hash = hash;
          for (uint64_t i = begin; i < end; ++i) {
            op.dispatches.push_back(static_cast<uint32_t>(i));
          }
          db.ops.push_back(op);
        }
      }
    }
  }
  impl.compileAll(!options.quite);

  SymGraphEval symEval = symGraph.eval(specs);

  db.shaderBinaries = implModel.shaderBinaries;
  db.dispatches.reserve(implModel.dispatches.size());
  for (size_t i = 0; i < implModel.dispatches.size(); ++i) {
    const auto &dispatch = implModel.dispatches[i];
    DbComputeDispatch dbDispatch;
    dbDispatch.workgroupCountX =
        static_cast<uint32_t>(symEval[dispatch.workgroupCount[0]].value());
    dbDispatch.workgroupCountY =
        static_cast<uint32_t>(symEval[dispatch.workgroupCount[1]].value());
    dbDispatch.workgroupCountZ =
        static_cast<uint32_t>(symEval[dispatch.workgroupCount[2]].value());
    dbDispatch.binaryId = dispatch.binaryId;
    for (const TensorBinding &binding : dispatch.bindings) {
      DbTensorBinding dbbinding;
      dbbinding.set = binding.set;
      dbbinding.binding = binding.binding;
      dbbinding.access = binding.accessFlag;
      const TensorStorageRequirements &tensor =
          implModel.tensors[binding.tensorId.index];
      dbbinding.tensorByteSize =
          static_cast<uint64_t>(*symEval[tensor.byteSize]);
      dbbinding.tensorMinAlignment = tensor.minAlignment;
      dbDispatch.bindings.push_back(dbbinding);
    }

    uint64_t offset = 0;
    for (const PushConstant &pc : dispatch.pushConstants) {
      offset = algorithm::align_up(offset, pc.type().alignment());
      offset += pc.type().size();
    }
    memory::vector<uint8_t> pushConstant(offset);
    offset = 0;
    for (const PushConstant &pc : dispatch.pushConstants) {
      offset = algorithm::align_up(offset, pc.type().alignment());
      if (pc.isDynamic()) {
        int64_t v = *symEval[Sym::Symbol(pc.dynamic())];
        switch (pc.type().kind()) {
        case memory::DtypeKind::F16:
        case memory::DtypeKind::F32:
        case memory::DtypeKind::F64:
          diag::not_implemented(); // <- floating point push constants are not
                                   // implemented.
        case memory::DtypeKind::U32: {
          uint32_t x = static_cast<uint32_t>(v);
          std::memcpy(pushConstant.data() + offset, &x, sizeof(uint32_t));
          break;
        }
        case memory::DtypeKind::I32: {
          int32_t x = static_cast<int32_t>(v);
          std::memcpy(pushConstant.data() + offset, &x, sizeof(int32_t));
          break;
        }
        }
      } else {
        switch (pc.type().kind()) {
        case memory::DtypeKind::F16:
        case memory::DtypeKind::F32:
        case memory::DtypeKind::F64:
          diag::not_implemented(); // <- floating point push constants are not
                                   // implemented.
        case memory::DtypeKind::U32: {
          uint32_t x = pc.u32();
          std::memcpy(pushConstant.data() + offset, &x, sizeof(uint32_t));
          break;
        }
        case memory::DtypeKind::I32: {
          int32_t x = pc.i32();
          std::memcpy(pushConstant.data() + offset, &x, sizeof(int32_t));
          break;
        }
        }
      }
      offset += pc.type().size();
    }
    dbDispatch.pushConstant = std::move(pushConstant);
    db.dispatches.push_back(std::move(dbDispatch));
  }
  return db;
}
} // namespace denox::compiler
