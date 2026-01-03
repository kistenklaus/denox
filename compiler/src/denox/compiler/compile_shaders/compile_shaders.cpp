#include "denox/compiler/compile_shaders/compile_shaders.hpp"
#include "denox/common/SHA256.hpp"
#include "denox/compiler/Options.hpp"
#include "denox/compiler/frontend/model/Model.hpp"
#include "denox/db/Db.hpp"
#include "denox/db/DbTensorBinding.hpp"
#include "denox/glsl/GlslCompilerInstance.hpp"
#include "denox/memory/container/hashmap.hpp"
#include <fmt/base.h>
#include <fmt/ostream.h>
#include <limits>

namespace denox::compiler {

static constexpr uint32_t u32sential = std::numeric_limits<uint32_t>::max();

SpvSchedule compile_shaders(MemSchedule &&schedule, const Model &model, Db &db,
                            [[maybe_unused]] spirv::GlslCompiler *glslCompiler,
                            const Options &options) {
  assert(glslCompiler != nullptr); // only to make lifetime intent visible

  memory::vector<SpvDispatch> dispatches;
  dispatches.reserve(schedule.dispatches.size());

  struct GlslTarget {
    spirv::GlslCompilerInstance glsl;
    memory::vector<size_t> dispatches;
  };

  memory::hash_map<SHA256, GlslTarget> targets;

  // Collect unique shader binaries.
  for (size_t i = 0; i < schedule.dispatches.size(); ++i) {
    auto &dispatch = schedule.dispatches[i];
    SHA256Builder hasher;
    dispatch.glsl.sha256(hasher);
    SHA256 hash = hasher.finalize();
    if (targets.contains(hash)) {
      targets.at(hash).dispatches.push_back(i);
    } else {
      targets.emplace(hash, //
                      GlslTarget{
                          .glsl = std::move(dispatch.glsl),
                          .dispatches = {i},
                      });
    }
    dispatches.push_back(SpvDispatch{
        .binaryId = u32sential,
        .pushConstants = dispatch.pushConstants,
        .workgroupCountX = dispatch.workgroupCountX,
        .workgroupCountY = dispatch.workgroupCountY,
        .workgroupCountZ = dispatch.workgroupCountZ,
        .bindings = dispatch.bindings,
        .info = dispatch.info,
    });
  }

  // eval symgraph with assumed dimensions!
  memory::small_vector<SymSpec, 4> symSpecs;
  for (const auto &assumption : options.assumptions.valueAssumptions) {
    auto it = std::ranges::find_if(
        model.valueNames(), [&](const NamedValue &namedValue) {
          return namedValue.name == assumption.valueName;
        });
    if (it == model.valueNames().end()) {
      continue;
    }
    auto namedValue = *it;
    if (namedValue.value.isConstant()) {
      continue;
    }
    Sym::symbol symbol = namedValue.value.sym();
    int64_t value = static_cast<int64_t>(assumption.value);
    symSpecs.push_back(SymSpec{.symbol = symbol, .value = value});
  }
  SymGraphEval eval = schedule.symGraph.eval(symSpecs);

  // collect binaries.
  memory::vector<SpirvBinary> binaries(targets.size());

  struct GlslCompilationUnit {
    size_t binaryId;
    SHA256 hash;
    spirv::GlslCompilerInstance glsl;
    memory::span<const size_t> dispatches;
  };
  memory::vector<GlslCompilationUnit> units;

  size_t i = 0;
  for (const auto &[hash, target] : targets) {
    memory::optional<SpirvBinary> cachedBinary = db.query_shader_binary(hash);
    if (cachedBinary.has_value()) {
      binaries[i] = *cachedBinary;
    } else {
      units.push_back(GlslCompilationUnit{
          .binaryId = i,
          .hash = hash,
          .glsl = target.glsl,
          .dispatches = target.dispatches,
      });
    }
    for (const size_t d : target.dispatches) {
      dispatches[d].binaryId = static_cast<uint32_t>(i);
    }
    ++i;
  }

  for (size_t u = 0; u < units.size(); ++u) {
    auto &unit = units[u];
    const uint32_t percentage =
        50 +
        static_cast<uint32_t>(std::floor(static_cast<float>(u + 1) * 40.0f /
                                         static_cast<float>(units.size())));
    fmt::println("[{:>3}%] \x1B[32mBuilding SPIR-V compute shader {}\x1B[0m",
                 percentage,
                 unit.glsl.getSourcePath().relative_to(io::Path::cwd()));

    SpirvBinary binary = *unit.glsl.compile();
    [[maybe_unused]] const bool inserted //
        = db.insert_binary(unit.hash, binary);
    assert(inserted);
    binaries[unit.binaryId] = std::move(binary);
  }

  return SpvSchedule{
      .symGraph = std::move(schedule.symGraph),
      .tensors = std::move(schedule.tensors),
      .buffers = std::move(schedule.buffers),
      .initializers = std::move(schedule.initializers),
      .dispatches = std::move(dispatches),
      .binaries = std::move(binaries),
      .inputs = std::move(schedule.inputs),
      .outputs = std::move(schedule.outputs),
  };
}

} // namespace denox::compiler
