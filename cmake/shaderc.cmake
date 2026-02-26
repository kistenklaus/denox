include_guard(GLOBAL)
include(FetchContent)

FetchContent_Declare(
    shaderc_known_good
    GIT_REPOSITORY https://github.com/google/shaderc.git
    GIT_TAG        3108af8ce49ae67066bb9289ed889a85b0c3e393 #v2026.1
)


FetchContent_MakeAvailable(shaderc_known_good)
