#include "denox/common/TensorFormat.hpp"
#include "denox/compiler/Options.hpp"
#include "denox/compiler/canonicalize/canonicalize.hpp"
#include "denox/compiler/compile.hpp"
#include "denox/compiler/compile_shaders/compile_shaders.hpp"
#include "denox/compiler/compile_symbols/compile_symbols.hpp"
#include "denox/compiler/dce/dce.hpp"
#include "denox/compiler/frontend/frontend.hpp"
#include "denox/compiler/implement/Supergraph.hpp"
#include "denox/compiler/implement/implement.hpp"
#include "denox/compiler/lifeness/lifeness.hpp"
#include "denox/compiler/placement/placement.hpp"
#include "denox/compiler/populate.hpp"
#include "denox/compiler/rebind_descriptors/rebind_descriptors.hpp"
#include "denox/compiler/selection/selection.hpp"
#include "denox/compiler/serialize/serialize.hpp"
#include "denox/compiler/specialization/specialization.hpp"
#include "denox/device_info/ApiVersion.hpp"
#include "denox/device_info/query/query_driver_device_info.hpp"
#include "denox/glsl/GlslCompiler.hpp"
#include "denox/io/fs/File.hpp"
#include "denox/io/fs/Path.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/memory/tensor/ActivationDescriptor.hpp"
#include "denox/memory/tensor/ActivationShape.hpp"
#include "denox/memory/tensor/ActivationTensor.hpp"
#include "denox/runtime/context.hpp"
#include "denox/runtime/db.hpp"
#include "denox/runtime/instance.hpp"
#include "denox/runtime/model.hpp"
#include "denox/spirv/ShaderDebugInfoLevel.hpp"
#include "denox/spirv/SpirvTools.hpp"
#include <fmt/base.h>
#include <fmt/format.h>

using namespace denox;
using namespace denox::io;

int main() {

  File file = File::open(Path("./net.onnx"), File::OpenMode::Read);
  std::vector<std::byte> onnx(file.size());
  file.read_exact(onnx);

  compiler::CompileOptions options{};
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

  uint32_t H = 1080;
  uint32_t W = 1920;
  uint32_t C = 3;

  auto context =  runtime::Context::make("*RTX*", ApiVersion::VULKAN_1_4);;

  auto db = Db::open(io::Path::cwd() / "gpu.db");
  {
    auto dnxbuf = denox::compile(onnx, db, options);
    auto model = denox::runtime::Model::make(dnxbuf, context);
    auto instance = denox::runtime::Instance::make(model, {{"H", H}, {"W", W}});
    fmt::println("{}", instance->bench().report());
  }

  denox::populate(db, onnx, options);

  db.atomic_writeback();

  auto rdb = denox::runtime::Db::open(context, db);
  rdb->bench();

  auto dnxbuf = denox::compile(onnx, db, options);
  auto model = denox::runtime::Model::make(dnxbuf, context);
  auto instance = denox::runtime::Instance::make(model, {{"H", H}, {"W", W}});

  fmt::println("{}", instance->bench().report());
  // memory::ActivationTensor inputTensor{
  //     memory::ActivationDescriptor{
  //         {W, H, C}, memory::ActivationLayout::HWC, memory::Dtype::F16},
  // };
  //
  // void *inputBuf = inputTensor.data();
  // std::memset(inputBuf, 0x3C, inputTensor.byteSize());
  // void **pInputs = &inputBuf;
  //
  // memory::ActivationTensor outputTensor{
  //     memory::ActivationDescriptor{
  //         {W, H, C}, memory::ActivationLayout::HWC, memory::Dtype::F16},
  // };
  //
  // void *outputBuf = outputTensor.data();
  // void **pOutputs = &outputBuf;
  //
  // instance->infer(pInputs, pOutputs);
  //
  // for (uint32_t c = 0; c < C; ++c) {
  //   fmt::println("CHANNEL: {}", c);
  //   for (uint32_t h = 0; h < H; ++h) {
  //     for (uint32_t w = 0; w < W; ++w) {
  //       fmt::print(" {:^7.3f} ", static_cast<float>(outputTensor.at(w, h,
  //       c)));
  //     }
  //     fmt::println("");
  //   }
  // }

  // fmt::println("{}", instance->bench().report());
}
