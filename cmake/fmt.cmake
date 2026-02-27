include_guard(GLOBAL)  

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG        e69e5f977d458f2650bb346dadf2ad30c5320281 # 10.2.1
  GIT_SHALLOW TRUE
  OVERRIDE_FIND_PACKAGE
  GIT_PROGRESS TRUE
) 

FetchContent_MakeAvailable(fmt)


add_library(denox_fmt INTERFACE)
target_link_libraries(denox_fmt INTERFACE fmt)
target_include_directories(denox_fmt 
  SYSTEM INTERFACE $<TARGET_PROPERTY:fmt,INTERFACE_INCLUDE_DIRECTORIES>
)


