#pragma once

#include "parser/artefact.hpp"
#include "parser/lex/tokens/token.hpp"
#include <cstdint>
#include <span>
#include <stdexcept>

std::optional<Artefact> parse_artefact(const Token &token);
