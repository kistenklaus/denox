

enum class HelpScope {
  Global,
  Bench,
  Compile,
  Infer,
  Populate,
  Reweight,
};

struct HelpAction {
  HelpScope scope;
};
