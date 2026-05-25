#pragma once

#include "denox/cli/parser/artefact.hpp"
#include "denox/compiler/Options.hpp"
#include "denox/memory/container/optional.hpp"

struct PopulateAction {
  OnnxArtefact model;
  DbArtefact database;

  // device info query
  std::variant<std::monostate, denox::memory::string,
    IOEndpoint> device;
  denox::ApiVersion apiVersion;

  // compile options
  denox::compiler::CompileOptions options;

  denox::diag::LogLevel loglevel = denox::diag::LogLevel::Info;
  bool logcolors = true;
};
