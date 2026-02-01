#pragma once

#include "denox/cli/parser/action.hpp"
#include "denox/cli/parser/lex/tokens/token.hpp"
#include <span>

Action parse_compile(std::span<const Token> tokens);

Action parse_populate(std::span<const Token> tokens);

Action parse_bench(std::span<const Token> tokens);

Action parse_infer(std::span<const Token> tokens);

Action parse_version(std::span<const Token> tokens);

Action parse_help(std::span<const Token> tokens);

Action parse_dumpcsv(std::span<const Token> tokens);
