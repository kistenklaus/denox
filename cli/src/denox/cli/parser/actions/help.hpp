

enum class HelpScope {
  Global,
  Bench,
  Compile,
  Infer,
  Populate,
  Reweight,
  QueryDeviceInfo,
  MergeDeviceInfo,
};

struct HelpAction {
  HelpScope scope;
};
