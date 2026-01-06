#pragma once

#include "denox/cli/parser/artefact.hpp"
#include "denox/compiler/Options.hpp"

struct PopulateAction  {
  OnnxArtefact model;
  DbArtefact database;


  // device info query
  denox::memory::optional<denox::memory::string> deviceName;
  denox::ApiVersion apiVersion;

  // compile options
  denox::compiler::CompileOptions options;
};
