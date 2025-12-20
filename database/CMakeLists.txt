
include(${PROJECT_SOURCE_DIR}/cmake/db.cmake)


add_library(denox_database 
  EXCLUDE_FROM_ALL
  src/denox/db/Db.cpp
)

target_compile_features(denox_database PUBLIC cxx_std_20)
set_target_properties(denox_database PROPERTIES CXX_STANDARD 20)
set_target_properties(denox_database PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(denox_database PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(denox_database 
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)


target_link_libraries(denox_database
  PUBLIC
    denox_common
    denox_spirv
    denox::db
)

if (NOT DENOX_DISABLE_WARNINGS)
  denox_enable_strict_warnings(denox_database)
endif()

if (DENOX_SAN) 
  target_compile_options(denox_database PRIVATE -fsanitize=undefined -fsanitize=address)
  target_link_options(denox_database PRIVATE -fsanitize=undefined -fsanitize=address)
endif()
