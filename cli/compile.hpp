#pragma once

#include "denox/compiler.hpp"
#include "parser/artefact.hpp"
#include <optional>

void compile(OnnxArtefact onnx, IOEndpoint output,
             std::optional<DbArtefact> database, denox::CompileOptions);
