#include "alloc/monotone_alloc.hpp"
#include "denox/cli/bench.hpp"
#include "denox/cli/compile.hpp"
#include "denox/cli/help.hpp"
#include "denox/cli/infer.hpp"
#include "denox/cli/parser/parse.hpp"
#include "denox/cli/populate.hpp"
#include "denox/cli/version.hpp"
#include "denox/diag/not_implemented.hpp"
#include "denox/io/fs/Path.hpp"
#include <fmt/ostream.h>
#include <fmt/printf.h>
#include <iostream>

int main(int argc, char **argv) {
  try {
    auto action = parse_argv(argc, argv);
    switch (action.kind()) {
    case ActionKind::Compile:
      compile(action.compile());
      break;
    case ActionKind::Infer:
      infer(action.infer());
      break;
    case ActionKind::Bench:
      bench(action.bench());
      break;
    case ActionKind::Populate:
      populate(action.populate());
      break;
    case ActionKind::Help:
      help(action.help());
      break;
    case ActionKind::Version:
      version();
      break;
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
  }

  mono_free_all();
}
