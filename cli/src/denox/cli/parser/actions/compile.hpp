#pragma once

#include "denox/cli/io/IOEndpoint.hpp"
#include "denox/cli/parser/artefact.hpp"
#include "denox/compiler/Options.hpp"
#include <optional>

struct CompileAction {
  OnnxArtefact input; // must exist
  IOEndpoint output;

  // device info query
  denox::memory::optional<denox::memory::string> deviceName;
  denox::ApiVersion apiVersion;

  // compile options.
  std::optional<DbArtefact> database;
  denox::compiler::CompileOptions options;
};
