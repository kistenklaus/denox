

enum class HelpScope {
  Global,
  Bench,
  Compile,
  Infer,
  Populate,
  Reweight,
  QueryDeviceInfo,
  MergeDeviceInfo,
  Dump,
};

struct HelpAction {
  HelpScope scope;
};
