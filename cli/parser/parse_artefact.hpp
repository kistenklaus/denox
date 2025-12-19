#pragma once

#include "parser/artefact.hpp"
#include "parser/lex/tokens/token.hpp"
#include <cstdint>
#include <span>
#include <stdexcept>

enum class ArtefactParseError {
  NotAnArtefactToken,
  PathDoesNotExist,
  UnrecognizedFormat,
  DatabasePiped,
  AmbiguousFormat,
};

struct ArtefactParseResult {
  std::optional<Artefact> artefact;
  std::optional<ArtefactParseError> error;

  static ArtefactParseResult success(Artefact a) {
    return {.artefact = std::move(a), .error = std::nullopt};
  }

  static ArtefactParseResult failure(ArtefactParseError e) {
    return {.artefact = std::nullopt, .error = e};
  }
};

bool is_onnx_model(std::span<const std::byte> data);

bool is_dnx_model(std::span<const std::byte> data);
bool is_db(std::span<const std::byte> data);

ArtefactParseResult parse_artefact(const Token &token);
