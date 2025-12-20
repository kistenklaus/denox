include_guard(GLOBAL)

include(${PROJECT_SOURCE_DIR}/cmake/flatbuffers.cmake)

denox_add_fbs_lib(denox::dnx
  ${PROJECT_SOURCE_DIR}/dnx.fbs
)
