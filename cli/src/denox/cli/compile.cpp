#include "denox/cli/compile.hpp"
#include "denox/compiler/compile.hpp"
#include "denox/device_info/query/query_driver_device_info.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/io/fs/File.hpp"

void compile(CompileAction &action) {
  denox::ApiVersion apiVersion = action.apiVersion;
  action.options.deviceInfo =
      denox::query_driver_device_info(apiVersion, action.deviceName);

  denox::memory::optional<denox::Db> db;
  if (action.database) {
    if (action.database->endpoint.kind() != IOEndpointKind::Path) {
      denox::diag::invalid_state();
    }
    db = denox::Db::open(action.database->endpoint.path());
  }

  action.options.assumptions.valueAssumptions.emplace_back("H", 1080);
  action.options.assumptions.valueAssumptions.emplace_back("W", 1920);
  action.options.debugInfo = denox::compiler::DebugInfo::Enable;

  auto dnxbuf = denox::compile(action.input.data, db, action.options);

  switch (action.output.kind()) {
  case IOEndpointKind::Path: {
    auto outfile = denox::io::File::open(
        action.output.path(), denox::io::File::OpenMode::Create |
                                  denox::io::File::OpenMode::Write |
                                  denox::io::File::OpenMode::Truncate);
    outfile.write_exact(dnxbuf);
    break;
  }
  case IOEndpointKind::Pipe:
    Pipe{}.write_exact(dnxbuf);
  }
}
