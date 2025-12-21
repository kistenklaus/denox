#include "denox/common/TensorFormat.hpp"
#include "denox/compiler/Options.hpp"
#include "denox/compiler/canonicalize/canonicalize.hpp"
#include "denox/compiler/dce/dce.hpp"
#include "denox/compiler/frontend/frontend.hpp"
#include "denox/compiler/lifeness/lifeness.hpp"
#include "denox/compiler/specialization/specialization.hpp"
#include "denox/io/fs/File.hpp"
#include "denox/io/fs/Path.hpp"
#include <fmt/format.h>

using namespace denox;
using namespace denox::io;

int main() {

  File file = File::open(Path("./net.onnx"), File::OpenMode::Read);
  std::vector<std::byte> onnx(file.size());
  file.read_exact(onnx);

  compiler::Options options;

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
  output.format = TensorFormat::TEX_RGBA;
  output.storage = TensorStorage::StorageImage;
  output.dtype = TensorDataType::Float16;

  options.interfaceDescriptors = {albedo, norm, output};

  compiler::Model model = compiler::frontend(onnx, options);
  fmt::println("MODEL:\n{}", model);

  compiler::CanoModel cano = compiler::canonicalize(model);

  fmt::println("CANO:\n{}", cano);

  compiler::Lifetimes lifetimes = compiler::lifeness(cano);

  compiler::SpecModel specModel = compiler::specialize(cano, lifetimes);

  fmt::println("SPEC:\n{}", specModel);

  compiler::ConstModel constModel = compiler::dce(specModel);

  fmt::println("DCE:\n{}", constModel);
}
