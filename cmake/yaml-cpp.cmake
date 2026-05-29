include_guard(GLOBAL) 

include(FetchContent)

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG 56e3bb550c91fd7005566f19c079cb7a503223cf
  GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(yaml-cpp)
