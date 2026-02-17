include_guard(GLOBAL)
include(${PROJECT_SOURCE_DIR}/cmake/colorful.cmake)

# Options to control how we get SQLite
option(DENOX_VENDOR_SQLITE3 "Download/build SQLite if not found" ON)
option(DENOX_SQLITE3_BUILD_SHARED "Build vendored SQLite as a shared library" OFF)

# Common SQLite feature toggles (affect vendored build only)
option(DENOX_SQLITE3_ENABLE_FTS5 "Enable FTS5 full-text search" ON)
option(DENOX_SQLITE3_ENABLE_RTREE "Enable R*Tree index extension" ON)
option(DENOX_SQLITE3_ENABLE_JSON1 "Enable JSON1 (mostly relevant for older SQLite)" ON)
option(DENOX_SQLITE3_ENABLE_LOAD_EXTENSION "Allow runtime loadable extensions (dlopen/LoadLibrary)" OFF)

# System lookup constraint
set(DENOX_SQLITE3_MIN_VERSION "3.35" CACHE STRING "Minimum SQLite version")

# Vendoring defaults (amalgamation zip)
# NOTE: SQLite encodes versions in filenames. Example: 3.51.2 -> 3510200
set(DENOX_SQLITE3_AMALGAMATION_ID "3510200" CACHE STRING "SQLite amalgamation file id (e.g. 3510200 for 3.51.2)")
set(DENOX_SQLITE3_DOWNLOAD_YEAR "2026" CACHE STRING "SQLite download year folder (used for the default URL)")
set(DENOX_SQLITE3_URL
    "https://www.sqlite.org/${DENOX_SQLITE3_DOWNLOAD_YEAR}/sqlite-amalgamation-${DENOX_SQLITE3_AMALGAMATION_ID}.zip"
    CACHE STRING "SQLite amalgamation URL (override if needed)"
)

# --- 1) Try system package first (CONFIG then MODULE) ---
# Some package managers provide a config package; otherwise CMake's FindSQLite3 module is typical.
find_package(SQLite3 ${DENOX_SQLITE3_MIN_VERSION} CONFIG QUIET)
if (NOT SQLite3_FOUND)
  find_package(SQLite3 ${DENOX_SQLITE3_MIN_VERSION} QUIET)
endif()

if (SQLite3_FOUND)
  # CMake historically provided SQLite::SQLite3; newer CMake also uses SQLite3::SQLite3 (and may deprecate the former).
  if (TARGET SQLite3::SQLite3)
    log_success("✅ SQLite (${SQLite3_VERSION}) available (system): target SQLite3::SQLite3")
    add_library(denox::sqlite3 ALIAS SQLite3::SQLite3)
    return()
  elseif (TARGET SQLite::SQLite3)
    log_success("✅ SQLite (${SQLite3_VERSION}) available (system): target SQLite::SQLite3")
    add_library(denox::sqlite3 ALIAS SQLite::SQLite3)
    return()
  else()
    # Fallback if no imported target is provided
    log_success("✅ SQLite (${SQLite3_VERSION}) available (system): using include/libs variables")
    add_library(denox-sqlite3-system INTERFACE EXCLUDE_FROM_ALL)
    target_include_directories(denox-sqlite3-system INTERFACE ${SQLite3_INCLUDE_DIRS})
    target_link_libraries(denox-sqlite3-system INTERFACE ${SQLite3_LIBRARIES})
    add_library(denox::sqlite3 ALIAS denox-sqlite3-system)
    return()
  endif()
endif()

# --- 2) Vendor via amalgamation (sqlite3.c + sqlite3.h) ---
if (DENOX_VENDOR_SQLITE3)
  include(FetchContent)

  FetchContent_Declare(denox_sqlite3_amalgamation
    URL ${DENOX_SQLITE3_URL}
  )
  FetchContent_MakeAvailable(denox_sqlite3_amalgamation)

  set(_sqlite3_src "${denox_sqlite3_amalgamation_SOURCE_DIR}/sqlite3.c")
  set(_sqlite3_inc "${denox_sqlite3_amalgamation_SOURCE_DIR}")

  if (DENOX_SQLITE3_BUILD_SHARED)
    add_library(denox-sqlite3-impl SHARED ${_sqlite3_src})
    # On Windows, export all symbols automatically for a DLL build
    set_target_properties(denox-sqlite3-impl PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
  else()
    add_library(denox-sqlite3-impl STATIC ${_sqlite3_src})
    # Helps if you later link this static lib into a shared lib
    set_target_properties(denox-sqlite3-impl PROPERTIES POSITION_INDEPENDENT_CODE ON)
  endif()

  target_include_directories(denox-sqlite3-impl PUBLIC ${_sqlite3_inc})

  # Feature flags (vendored build only)
  target_compile_definitions(denox-sqlite3-impl PRIVATE SQLITE_THREADSAFE=1)

  if (DENOX_SQLITE3_ENABLE_FTS5)
    target_compile_definitions(denox-sqlite3-impl PRIVATE SQLITE_ENABLE_FTS5)
  endif()
  if (DENOX_SQLITE3_ENABLE_RTREE)
    target_compile_definitions(denox-sqlite3-impl PRIVATE SQLITE_ENABLE_RTREE)
  endif()
  if (DENOX_SQLITE3_ENABLE_JSON1)
    target_compile_definitions(denox-sqlite3-impl PRIVATE SQLITE_ENABLE_JSON1)
  endif()
  if (NOT DENOX_SQLITE3_ENABLE_LOAD_EXTENSION)
    target_compile_definitions(denox-sqlite3-impl PRIVATE SQLITE_OMIT_LOAD_EXTENSION)
  endif()

  # Link requirements (mostly relevant when loadable extensions / threading are enabled)
  find_package(Threads QUIET)
  if (Threads_FOUND)
    target_link_libraries(denox-sqlite3-impl PRIVATE Threads::Threads)
  endif()
  if (DENOX_SQLITE3_ENABLE_LOAD_EXTENSION)
    target_link_libraries(denox-sqlite3-impl PRIVATE ${CMAKE_DL_LIBS})
  endif()

  # Public wrapper target (mirrors your fmt pattern)
  add_library(denox-sqlite3 INTERFACE EXCLUDE_FROM_ALL)
  target_link_libraries(denox-sqlite3 INTERFACE denox-sqlite3-impl)

  log_success("✅ SQLite (vendored amalgamation): ${DENOX_SQLITE3_URL}")
  add_library(denox::sqlite3 ALIAS denox-sqlite3)
  return()
endif()

log_error("❌ SQLite3 not available! (system not found and DENOX_VENDOR_SQLITE3=OFF)")
