include_guard(GLOBAL)
find_package(SQLite3 REQUIRED)

if(NOT TARGET SQLite::SQLite3)
  message(FATAL_ERROR "SQLite::SQLite3 imported target missing")
endif()
