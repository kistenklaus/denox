#pragma once

#include "denox/cli/parser/artefact.hpp"
#include "denox/common/ValueSpec.hpp"
#include "denox/compiler/Options.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/runtime/db.hpp"
#include <optional>

struct BenchAction {
  Artefact target;

  // device info query
  denox::memory::optional<denox::memory::string> deviceName;
  denox::ApiVersion apiVersion;

  // db bench options (only valid if target is a database)
  denox::runtime::DbBenchOptions benchOptions;

  // all following options are only valid if the target is a onnx artefact
  // compile options.
  std::optional<DbArtefact> database;
  denox::compiler::CompileOptions options;

  // dnx bench spec. (only valid if target is a onnx or dnx artefact)
  denox::memory::vector<denox::ValueSpec> valueSpecs;
};
