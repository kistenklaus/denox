include_guard(GLOBAL)

include(cmake/flatbuffers.cmake)

denox_add_fbs_lib(denox::db
  ${CMAKE_CURRENT_SOURCE_DIR}/db.fbs
)
