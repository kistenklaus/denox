#include "compile.hpp"
#include "alloc/monotone_alloc.hpp"
#include "denox/compiler.hpp"
#include "io/File.hpp"
#include "io/IOEndpoint.hpp"
#include <stdexcept>

void compile(OnnxArtefact onnx, IOEndpoint output,
             std::optional<DbArtefact> database,
             denox::CompileOptions options) {

  char *db = nullptr;
  if (database.has_value()) {
    const std::string dbpath = database->endpoint.path().str();
    db = static_cast<char *>(mono_alloc(dbpath.size() + 1));
    std::memset(db, 0, dbpath.size() + 1);
    std::memcpy(db, dbpath.data(), dbpath.size());
  }

  denox::CompilationResult result;
  int erno =
      denox::compile(onnx.data.data(), onnx.data.size(), &options, db, &result);
  if (erno) {
    throw std::runtime_error(result.message);
  }

  if (output.kind() == IOEndpointKind::Path) {
    File file = File::open(output.path(), File::OpenMode::Create |
                                              File::OpenMode::Write |
                                              File::OpenMode::Truncate);
    file.write_exact(
        std::span{static_cast<const std::byte *>(result.dnx), result.dnxSize});
  } else {
    assert(output.kind() == IOEndpointKind::Pipe);
    output.pipe().write_all(
        std::span{static_cast<const std::byte *>(result.dnx), result.dnxSize});
  }
}
