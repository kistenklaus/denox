#pragma once

namespace denox {

struct TypeSupport {
  bool storageBuffer8BitAccess;
  bool storageBuffer16BitAccess;
  bool float16;
  bool int8;
  bool globalInt64Atomic;
  bool sharedInt64Atomic;
  bool supportsFloatControl;
};

}
