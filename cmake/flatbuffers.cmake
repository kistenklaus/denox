include_guard(GLOBAL)
include(${PROJECT_SOURCE_DIR}/cmake/colorful.cmake)

# --- Wrapper you link against everywhere ---
if (NOT TARGET denox_flatbuffers)
  add_library(denox_flatbuffers INTERFACE)
  add_library(denox::flatbuffers ALIAS denox_flatbuffers)
endif()

# --- Expose flatc path AND target for build deps ---
set(DENOX_FLATC_EXECUTABLE "" CACHE STRING "Path/genex to flatc")
set(DENOX_FLATC_TARGET "" CACHE STRING "Target name for flatc")

# ============================================================
# System FlatBuffers
# ============================================================
if (DENOX_USE_SYSTEM_FLATBUFFERS)

  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FLATBUFFERS REQUIRED flatbuffers)

  # Create an interface target matching your abstraction
  target_include_directories(denox_flatbuffers SYSTEM INTERFACE
    ${FLATBUFFERS_INCLUDE_DIRS}
  )

  target_link_libraries(denox_flatbuffers INTERFACE
    ${FLATBUFFERS_LIBRARIES}
  )

  # flatc as a tool
  find_program(_flatc flatc REQUIRED)
  set(DENOX_FLATC_EXECUTABLE "${_flatc}" CACHE STRING "" FORCE)
  set(DENOX_FLATC_TARGET "" CACHE STRING "" FORCE)

  log_success("✅ flatbuffers available (system via pkg-config)")

# ============================================================
# FetchContent FlatBuffers
# ============================================================
else()

  # --- Fetch upstream exactly ---
  set(FLATBUFFERS_BUILD_FLATC     ON  CACHE BOOL "" FORCE)
  set(FLATBUFFERS_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
  set(FLATBUFFERS_INSTALL         OFF CACHE BOOL "" FORCE)
  set(FLATBUFFERS_BUILD_GRPCTEST  OFF CACHE BOOL "" FORCE)

  include(FetchContent)
  FetchContent_Declare(
    flatbuffers
    URL "https://github.com/google/flatbuffers/archive/refs/tags/v23.5.26.tar.gz"
    EXCLUDE_FROM_ALL
  )
  FetchContent_MakeAvailable(flatbuffers)

  # Link to whatever name upstream used
  if (TARGET flatbuffers::flatbuffers)
    target_link_libraries(denox_flatbuffers INTERFACE flatbuffers::flatbuffers)
  elseif (TARGET FlatBuffers::FlatBuffers)
    target_link_libraries(denox_flatbuffers INTERFACE FlatBuffers::FlatBuffers)
  elseif (TARGET flatbuffers) # legacy/plain
    target_link_libraries(denox_flatbuffers INTERFACE flatbuffers)
  else()
    log_error("FlatBuffers library target not found after fetch.")
  endif()

  # Normalize includes as SYSTEM
  get_target_property(_fb_includes denox_flatbuffers INTERFACE_INCLUDE_DIRECTORIES)
  if (_fb_includes)
    target_include_directories(denox_flatbuffers
      SYSTEM INTERFACE ${_fb_includes}
    )
  endif()

  # flatc as a target
  if (TARGET flatbuffers::flatc)
    set(DENOX_FLATC_EXECUTABLE "$<TARGET_FILE:flatbuffers::flatc>" CACHE STRING "" FORCE)
    set(DENOX_FLATC_TARGET "flatbuffers::flatc" CACHE STRING "" FORCE)
  elseif (TARGET flatc)
    set(DENOX_FLATC_EXECUTABLE "$<TARGET_FILE:flatc>" CACHE STRING "" FORCE)
    set(DENOX_FLATC_TARGET "flatc" CACHE STRING "" FORCE)
  else()
    log_error("flatc target not built (ensure FLATBUFFERS_BUILD_FLATC=ON).")
  endif()

  log_success("✅ flatbuffers available (FetchContent)")

endif()

# ---------- Helper: generate headers and wrap as an INTERFACE lib ----------
# denox_add_fbs_lib(<target_or_alias> <fbs1> [fbs2 ...]
#   [OUT_DIR <dir>]
#   [IMPORT_DIRS <dirs...>]
#   [IN_ALL]                  # build with 'all' (default OFF)
# )
function(denox_add_fbs_lib tgt)
  if (ARGC LESS 2)
    log_error("denox_add_fbs_lib(${tgt}): at least one .fbs file is required")
  endif()



  # Gather positional .fbs until we hit a keyword
  set(_args ${ARGV})
  list(POP_FRONT _args _tgt_in)
  set(_known OUT_DIR IMPORT_DIRS IN_ALL)
  set(_fbs)
  while(_args)
    list(GET _args 0 tok)
    list(FIND _known "${tok}" idx)
    if (idx GREATER -1)
      break()
    endif()
    list(APPEND _fbs "${tok}")
    list(POP_FRONT _args)
  endwhile()

  set(options IN_ALL)
  set(oneValue OUT_DIR)
  set(multiValue IMPORT_DIRS)
  cmake_parse_arguments(FBS "${options}" "${oneValue}" "${multiValue}" ${_args})



  # Resolve real target name (allow 'ns::alias' form)
  set(_tgt_real "${_tgt_in}")
  string(REPLACE "::" "_" _tgt_sanitized "${_tgt_in}")
  string(REGEX REPLACE "[^A-Za-z0-9_]" "_" _tgt_sanitized "${_tgt_sanitized}")
  if (_tgt_in MATCHES "::")
    set(_tgt_real "${_tgt_sanitized}")
  endif()

  if (NOT FBS_OUT_DIR)
    set(FBS_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/${_tgt_sanitized}")
  endif()
  file(MAKE_DIRECTORY "${FBS_OUT_DIR}")

  # Default import dirs: parents of the .fbs files
  if (NOT FBS_IMPORT_DIRS)
    set(_import_dirs)
    foreach(p IN LISTS _fbs)
      get_filename_component(_pdir "${p}" DIRECTORY)
      list(APPEND _import_dirs "${_pdir}")
    endforeach()
    list(REMOVE_DUPLICATES _import_dirs)
    set(FBS_IMPORT_DIRS "${_import_dirs}")
  endif()

  # Build flatc include args
  set(_flatc_I)
  foreach(dir IN LISTS FBS_IMPORT_DIRS)
    list(APPEND _flatc_I "-I" "${dir}")
  endforeach()

  list(JOIN _fbs " " _fbs_join)
  # log_info("flatc ${_fbs_join} >> ${FBS_OUT_DIR}")


  # One output header per .fbs
  set(_outs)
  foreach(p IN LISTS _fbs)
    get_filename_component(_name "${p}" NAME_WE)
    set(_out "${FBS_OUT_DIR}/${_name}.h")
    add_custom_command(
      OUTPUT "${_out}"
      COMMAND "${DENOX_FLATC_EXECUTABLE}" --cpp --no-emit-min-max-enum-values --filename-suffix "" -o "${FBS_OUT_DIR}" ${_flatc_I} "${p}" 
      DEPENDS "${p}"
      VERBATIM
    )
    list(APPEND _outs "${_out}")
  endforeach()

  # Collect under a custom target
  set(_gen_tgt "${_tgt_real}_fbs")
  add_custom_target("${_gen_tgt}" DEPENDS ${_outs})
  if (FBS_IN_ALL)
    add_dependencies(all "${_gen_tgt}")
  endif()


  # Header-only lib exposing generated includes
  add_library(${_tgt_real} INTERFACE)
  add_dependencies(${_tgt_real} "${_gen_tgt}")
  target_include_directories(${_tgt_real} SYSTEM INTERFACE "${FBS_OUT_DIR}")
  target_link_libraries(${_tgt_real} INTERFACE denox::flatbuffers)
  set_target_properties(${_tgt_real} PROPERTIES POSITION_INDEPENDENT_CODE BUILD_PIL)

  # Create requested alias if namespaced
  if (_tgt_in MATCHES "::")
    add_library(${_tgt_in} ALIAS ${_tgt_real})
  endif()


  # Log
  log_success("✅ Generated ${_tgt_in}")
endfunction()
