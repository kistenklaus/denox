#pragma once

#include "denox/memory/container/optional.hpp"
#include "denox/cli/io/IOEndpoint.hpp"
#include "denox/cli/parser/artefact.hpp"
#include "denox/compiler/Options.hpp"
#include <optional>
#include <variant>

struct CompileAction {
  OnnxArtefact input; // must exist
  IOEndpoint output;

  // device info query
  std::variant<std::monostate, denox::memory::string, IOEndpoint> device;
  denox::ApiVersion apiVersion;

  // compile options.
  std::optional<DbArtefact> database;
  denox::compiler::CompileOptions options;

  denox::diag::LogLevel loglevel = denox::diag::LogLevel::Info;
  bool logcolors = true;
};
