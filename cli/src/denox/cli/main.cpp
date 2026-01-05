

#include "alloc/monotone_alloc.hpp"
#include "denox/cli/compile.hpp"
#include "denox/cli/parser/parse.hpp"
#include "denox/diag/not_implemented.hpp"
#include <fmt/base.h>
#include <fmt/printf.h>
#include <stdexcept>

int main(int argc, char **argv) {

  auto action = parse_argv(argc, argv);

  switch (action.kind()) {
  case ActionKind::Compile:
    compile(action.compile());
    break;
  case ActionKind::Infer:
  case ActionKind::Bench:
  case ActionKind::Populate:
  case ActionKind::Help:
  case ActionKind::Version:
    denox::diag::not_implemented();
    break;
  }

  mono_free_all();
}
