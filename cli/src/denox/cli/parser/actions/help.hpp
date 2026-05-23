

enum class HelpScope {
  Global,
  Bench,
  Compile,
  Infer,
  Populate,
  Reweight,
  QueryDeviceInfo,
};

struct HelpAction {
  HelpScope scope;
};
