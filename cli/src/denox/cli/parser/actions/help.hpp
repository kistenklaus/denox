

enum class HelpScope {
  Global,
  Bench,
  Compile,
  Infer,
  Populate,
};

struct HelpAction {
  HelpScope scope;
};
