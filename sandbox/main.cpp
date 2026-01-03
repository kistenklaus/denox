#include "denox/common/TensorFormat.hpp"
#include "denox/compiler/Options.hpp"
#include "denox/compiler/canonicalize/canonicalize.hpp"
#include "denox/compiler/compile_shaders/compile_shaders.hpp"
#include "denox/compiler/compile_symbols/compile_symbols.hpp"
#include "denox/compiler/dce/dce.hpp"
#include "denox/compiler/frontend/frontend.hpp"
#include "denox/compiler/implement/Supergraph.hpp"
#include "denox/compiler/implement/implement.hpp"
#include "denox/compiler/lifeness/lifeness.hpp"
#include "denox/compiler/placement/placement.hpp"
#include "denox/compiler/rebind_descriptors/rebind_descriptors.hpp"
#include "denox/compiler/selection/selection.hpp"
#include "denox/compiler/specialization/specialization.hpp"
#include "denox/device_info/query/query_driver_device_info.hpp"
#include "denox/glsl/GlslCompiler.hpp"
#include "denox/io/fs/File.hpp"
#include "denox/io/fs/Path.hpp"
#include "denox/spirv/ShaderDebugInfoLevel.hpp"
#include "denox/spirv/SpirvTools.hpp"
#include <chrono>
#include <fmt/base.h>
#include <fmt/format.h>

using namespace denox;
using namespace denox::io;

int main() {

  File file = File::open(Path("./net.onnx"), File::OpenMode::Read);
  std::vector<std::byte> onnx(file.size());
  file.read_exact(onnx);

  compiler::Options options{};
  ApiVersion env = ApiVersion::VULKAN_1_4;
  DeviceInfo deviceInfo = denox::query_driver_device_info(env, memory::nullopt);
  options.deviceInfo = deviceInfo;

  compiler::InterfaceTensorDescriptor input;
  input.name = "input";
  input.format = TensorFormat::Optimal;
  input.storage = TensorStorage::Optimal;
  input.dtype = TensorDataType::Float16;
  input.widthValueName = "W";
  input.heightValueName = "H";

  compiler::InterfaceTensorDescriptor albedo;
  albedo.name = "albedo";
  albedo.format = TensorFormat::Optimal;
  albedo.storage = TensorStorage::Optimal;
  albedo.dtype = TensorDataType::Float16;
  albedo.widthValueName = "W";
  albedo.heightValueName = "H";

  compiler::InterfaceTensorDescriptor norm;
  norm.name = "norm";
  norm.format = TensorFormat::Optimal;
  norm.storage = TensorStorage::Optimal;
  norm.dtype = TensorDataType::Float16;
  norm.widthValueName = "W";
  norm.heightValueName = "H";

  compiler::InterfaceTensorDescriptor output;
  output.name = "output";
  output.format = TensorFormat::Optimal;
  output.storage = TensorStorage::Optimal;
  output.dtype = TensorDataType::Float16;

  options.interfaceDescriptors = {input, output};

  options.assumptions.valueAssumptions.emplace_back("H", 1080);
  options.assumptions.valueAssumptions.emplace_back("W", 1920);

  options.debugInfo = denox::compiler::DebugInfo::Enable;

  while (true) {

    spirv::SpirvTools tools(options.deviceInfo);
    spirv::GlslCompiler glslCompiler(&tools, options.deviceInfo,
                                     spirv::SpirvDebugInfoLevel::Strip);

    compiler::Model model = compiler::frontend(onnx, options);
    fmt::println("MODEL:\n{}", model);

    compiler::CanoModel cano = compiler::canonicalize(model);

    // fmt::println("CANO:\n{}", cano);

    compiler::Lifetimes lifetimes = compiler::lifeness(cano);

    compiler::SpecModel specModel = compiler::specialize(cano, lifetimes);

    // fmt::println("SPEC:\n{}", specModel);

    compiler::ConstModel constModel = compiler::dce(specModel);

    // fmt::println("DCE:\n{}", constModel);

    compiler::SuperGraph supergraph =
        compiler::implement(constModel, cano.symGraph, &glslCompiler, options);

    // fmt::println("SuperGraph:\n{}", supergraph);

    auto db = Db::open(io::Path::cwd() / "gpu.db");

    auto optschedule =
        compiler::select_schedule(std::move(supergraph), db, model, options);

    auto memschedule = compiler::placement(optschedule);

    auto schedule = compiler::compile_shaders(std::move(memschedule), model, db,
                                              &glslCompiler, options);

    compiler::rebind_descriptors(schedule, options, &tools);

    auto sprog = compiler::compile_symbols(schedule, model, options);

    db.atomic_writeback();

    fmt::println("[100%] \x1b[1m\x1B[32mSerialize dnx artefact ./net.dnx");
    break;
  }
}
