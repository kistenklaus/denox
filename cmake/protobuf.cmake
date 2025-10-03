# cmake/deps/protobuf.cmake
include_guard(GLOBAL)
include(cmake/colorful.cmake)

# --- Find Protobuf via the CMake module (needed for protobuf_generate_cpp)
if(APPLE)
  list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")
endif()

find_package(Protobuf CONFIG QUIET)
if(NOT Protobuf_FOUND)
  find_package(Protobuf QUIET)
endif()

if (NOT Protobuf_FOUND)
  log_error("❌ Protobuf not found. Install protoc & libprotobuf (set CMAKE_PREFIX_PATH if needed).")
  message(FATAL_ERROR "Protobuf is required.")
endif()

if (DEFINED Protobuf_VERSION)
  log_success("✅ Protobuf v${Protobuf_VERSION} found")
else()
  log_success("✅ Protobuf found")
endif()

# if (DEFINED Protobuf_PROTOC_EXECUTABLE)
#   log_info("protoc: ${Protobuf_PROTOC_EXECUTABLE}")
# endif()

# --- Create a stable wrapper you can link everywhere
add_library(denox::protobuf INTERFACE IMPORTED)

# Prefer the imported target if the module provides it; else use legacy vars
if (TARGET protobuf::libprotobuf)
  target_link_libraries(denox::protobuf INTERFACE protobuf::libprotobuf)
  # log_info("Linking protobuf::libprotobuf")
else()
  target_include_directories(denox::protobuf INTERFACE ${Protobuf_INCLUDE_DIRS})
  target_link_libraries(denox::protobuf INTERFACE ${Protobuf_LIBRARIES})
  info_success("Linking Protobuf via legacy variables")
endif()

# --------------------------------------------------------------------------
# Abseil only when needed (default: Debug builds). You can override the list.
# For single-config generators this avoids requiring absl for Release builds.
# For multi-config generators we try to find absl; if missing, Debug will fail.
# --------------------------------------------------------------------------
set(DENOX_PROTOBUF_ABSL_CONFIGS "Debug" CACHE STRING "Configs that must link Abseil with protobuf")


# Normalize config list
separate_arguments(DENOX_PROTOBUF_ABSL_CONFIGS)

if (CMAKE_CONFIGURATION_TYPES)
  # Multi-config generator (VS/Xcode): try to use absl if available
  find_package(absl CONFIG QUIET)
  if (absl_FOUND)
    # Link absl only for the listed configs
    foreach(_cfg IN LISTS DENOX_PROTOBUF_ABSL_CONFIGS)
      target_link_libraries(denox::protobuf INTERFACE
        $<$<CONFIG:${_cfg}>:absl::strings absl::cord absl::log absl::check>
      )
    endforeach()
    list(JOIN DENOX_PROTOBUF_ABSL_CONFIGS ", " _cfgs)
    log_info("✅ Linking absl to libprotobuf")
  else()
    list(JOIN DENOX_PROTOBUF_ABSL_CONFIGS ", " _cfgs)
    log_warn("⚠️ Abseil not found; build will fail if generated protobuf source requires Abseil.")
  endif()
else()
  # Single-config generator (Ninja/Make): only touch absl if current build type matches
  if (CMAKE_BUILD_TYPE IN_LIST DENOX_PROTOBUF_ABSL_CONFIGS)
    # Pull our absl wrapper (defines denox::absl or errors if truly missing)
    include(cmake/absl.cmake)
    target_link_libraries(denox::protobuf INTERFACE denox::absl)
    log_info("Linking absl to libprotobuf (Only a ${CMAKE_BUILD_TYPE} dependency)")
  endif()
endif()

# denox_add_proto_lib(<target_or_alias> <proto1> [proto2 ...]
#   [OUT_DIR <dir>]
#   [IMPORT_DIRS <dirs...>]
#   [TYPE STATIC|OBJECT]         # default STATIC
#   [CXX_STD <20|17|...>]        # default 20
#   [PIC <ON|OFF>]               # default ON
#   [IN_ALL]                     # opt-in: build with 'all' instead of EXCLUDE_FROM_ALL
# )
function(denox_add_proto_lib tgt)
  # ---- required: target + at least one .proto
  set(_args ${ARGV})
  list(POP_FRONT _args _tgt_in)
  if (NOT _args)
    log_error("❌ denox_add_proto_lib(${_tgt_in}): at least one .proto is required")
  endif()

  # Collect positional protos until we hit a known keyword
  set(_known OUT_DIR IMPORT_DIRS TYPE CXX_STD PIC IN_ALL)
  set(_protos)
  while(_args)
    list(GET _args 0 _tok)
    list(FIND _known "${_tok}" _idx)
    if (_idx GREATER -1)
      break()
    endif()
    list(APPEND _protos "${_tok}")
    list(POP_FRONT _args)
  endwhile()

  if (NOT _protos)
    log_error("❌ denox_add_proto_lib(${_tgt_in}): no .proto files were provided")
  endif()

  # Parse optional keywords
  set(options IN_ALL)
  set(oneValue OUT_DIR TYPE CXX_STD PIC)
  set(multiValue IMPORT_DIRS)
  cmake_parse_arguments(DAP "${options}" "${oneValue}" "${multiValue}" ${_args})

  # Defaults
  if (NOT DEFINED DAP_TYPE)
    set(DAP_TYPE STATIC)
  endif()
  if (NOT DEFINED DAP_CXX_STD)
    set(DAP_CXX_STD 20)
  endif()
  if (NOT DEFINED DAP_PIC)
    set(DAP_PIC ON)
  endif()

  # Sanitized internal target name if an alias like 'ns::name' was requested
  set(_tgt_real "${_tgt_in}")
  string(REPLACE "::" "_" _tgt_sanitized "${_tgt_in}")
  string(REGEX REPLACE "[^A-Za-z0-9_]" "_" _tgt_sanitized "${_tgt_sanitized}")
  if (_tgt_in MATCHES "::")
    set(_tgt_real "${_tgt_sanitized}")
  endif()

  # OUT_DIR default
  if (NOT DAP_OUT_DIR)
    set(DAP_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated/${_tgt_sanitized})
  endif()
  file(MAKE_DIRECTORY "${DAP_OUT_DIR}")

  # IMPORT_DIRS default: parent dirs of the protos (deduped)
  if (NOT DAP_IMPORT_DIRS)
    set(_import_dirs)
    foreach(p IN LISTS _protos)
      get_filename_component(_pdir "${p}" DIRECTORY)
      list(APPEND _import_dirs "${_pdir}")
    endforeach()
    list(REMOVE_DUPLICATES _import_dirs)
    set(DAP_IMPORT_DIRS "${_import_dirs}")
  endif()

  # Resolve protoc (support CONFIG target, MODULE var, or plain PATH)
  set(_PROTOC "")
  if (TARGET protobuf::protoc)
    # Use the path of the imported executable target
    set(_PROTOC "$<TARGET_FILE:protobuf::protoc>")
  elseif (DEFINED Protobuf_PROTOC_EXECUTABLE)
    set(_PROTOC "${Protobuf_PROTOC_EXECUTABLE}")
  else()
    find_program(_PROTOC NAMES protoc)
  endif()
  if (NOT _PROTOC)
    message(FATAL_ERROR "protoc not found. Ensure Protobuf is discovered (CONFIG or MODULE) or protoc is on PATH.")
  endif()

  # Construct protoc include args
  set(_proto_includes)
  foreach(dir IN LISTS DAP_IMPORT_DIRS)
    list(APPEND _proto_includes "-I" "${dir}")
  endforeach()

  # Gather generated files (mirror protoc's output relative to first matching -I root)
  set(_gen_srcs)
  set(_gen_hdrs)
  foreach(p IN LISTS _protos)
    # Compute path of p relative to the first IMPORT_DIR that contains it
    set(_rel "")
    foreach(_idir IN LISTS DAP_IMPORT_DIRS)
      file(RELATIVE_PATH _tmp_rel "${_idir}" "${p}")
      if (NOT _tmp_rel MATCHES "^[.][.]/") # inside this _idir
        set(_rel "${_tmp_rel}")
        break()
      endif()
    endforeach()
    if (_rel STREQUAL "")
      # Fallback: just use the filename
      get_filename_component(_rel "${p}" NAME)
    endif()

    # Strip extension; keep relative subdir
    get_filename_component(_rel_dir "${_rel}" DIRECTORY)
    get_filename_component(_rel_we  "${_rel}" NAME_WE)

    set(_out_dir "${DAP_OUT_DIR}")
    if (NOT _rel_dir STREQUAL "")
      set(_out_dir "${DAP_OUT_DIR}/${_rel_dir}")
    endif()

    set(_gen_src "${_out_dir}/${_rel_we}.pb.cc")
    set(_gen_hdr "${_out_dir}/${_rel_we}.pb.h")

    log_success("${CMAKE_COMMAND} -E make_directory ${_out_dir}")

    add_custom_command(
      OUTPUT "${_gen_src}" "${_gen_hdr}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${_out_dir}"
      COMMAND ${_PROTOC}
              --cpp_out=${DAP_OUT_DIR}
              ${_proto_includes}
              "${p}"
      DEPENDS "${p}"
      COMMENT "protoc ${p} -> ${_out_dir}"
      VERBATIM
    )

    list(APPEND _gen_srcs "${_gen_src}")
    list(APPEND _gen_hdrs "${_gen_hdr}")
  endforeach()

  if (NOT _gen_srcs)
    log_error("❌ Protobuf generation produced no sources for target '${_tgt_in}'")
  endif()

  # Define library
  add_library(${_tgt_real} ${DAP_TYPE} ${_gen_srcs} ${_gen_hdrs})

  if (NOT DAP_IN_ALL)
    set_property(TARGET ${_tgt_real} PROPERTY EXCLUDE_FROM_ALL TRUE)
  endif()
  if (DAP_PIC)
    set_property(TARGET ${_tgt_real} PROPERTY POSITION_INDEPENDENT_CODE ON)
  endif()

  target_compile_features(${_tgt_real} PUBLIC "cxx_std_${DAP_CXX_STD}")
  target_include_directories(${_tgt_real} PUBLIC "${DAP_OUT_DIR}")

  # Always route linking via your wrapper (handles absl per-config etc.)
  target_link_libraries(${_tgt_real} PUBLIC denox::protobuf)

  if (_tgt_in MATCHES "::")
    add_library(${_tgt_in} ALIAS ${_tgt_real})
  endif()
  log_success("✅ Generated target '${_tgt_in}'")
endfunction()
