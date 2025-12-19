

#include "alloc/monotone_alloc.hpp"
#include "compile.hpp"
#include "parser/parse.hpp"
#include <fmt/printf.h>

int main(int argc, char **argv) {

  std::optional<Action> actionOpt;
  try {
    actionOpt = parse_argv(argc, argv);
  } catch (const std::exception &e) {
    fmt::println("{}", e.what());
    return -1;
  }

  const Action &action = actionOpt.value();
  switch (action.kind()) {
  case ActionKind::Compile: {
    const auto &ac = action.compile();
    compile(ac.input, ac.output, ac.database, ac.options);
    break;
  }
  case ActionKind::Infer:
    break;
  case ActionKind::Bench:
    break;
  case ActionKind::Populate:
    break;
  case ActionKind::Help:
    break;
  case ActionKind::Version:
    break;
  }

    mono_free_all();
}
