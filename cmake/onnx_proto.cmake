# cmake/onnx_proto.cmake
# Generate ONNX protobuf sources into the build tree and make a library.
# This target is EXCLUDE_FROM_ALL; it builds only if you link it somewhere.

find_package(Protobuf REQUIRED)  # protoc + protobuf::libprotobuf

# ---- Config ----
# Root where your .proto(s) live
set(DENOX_ONNX_PROTO_ROOT "${PROJECT_SOURCE_DIR}/third_party" CACHE PATH "Root dir for ONNX protos")

# List of proto files. Default assumes a single file at third_party/onnx.proto.
# If you vendor the canonical pair, switch to the two lines below and ensure the subdirs exist.
set(DENOX_ONNX_PROTOS
  "${DENOX_ONNX_PROTO_ROOT}/onnx.proto"
  # "${DENOX_ONNX_PROTO_ROOT}/onnx/onnx.proto"
  # "${DENOX_ONNX_PROTO_ROOT}/onnx/onnx-ml.proto"
)

# Where generated .pb.cc/.pb.h go (in the build tree)
set(DENOX_ONNX_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/onnx")
file(MAKE_DIRECTORY "${DENOX_ONNX_GEN_DIR}")

# ---- Generate ----
protobuf_generate_cpp(DENOX_ONNX_SRCS DENOX_ONNX_HDRS
  ${DENOX_ONNX_PROTOS}
  PROTOC_OUT_DIR "${DENOX_ONNX_GEN_DIR}"
  IMPORT_DIRS    "${DENOX_ONNX_PROTO_ROOT}"
)

# ---- Library from generated code (excluded from ALL) ----
add_library(denox_onnx_proto STATIC
  ${DENOX_ONNX_SRCS}
  ${DENOX_ONNX_HDRS}
)
set_target_properties(denox_onnx_proto PROPERTIES
  EXCLUDE_FROM_ALL TRUE            # won't build unless linked or requested
  FOLDER "generated/onnx"          # IDE grouping (optional)
)

target_include_directories(denox_onnx_proto
  PUBLIC
    "${DENOX_ONNX_GEN_DIR}"
)

find_package(absl CONFIG QUIET)

target_link_libraries(denox_onnx_proto
  PUBLIC
    protobuf::libprotobuf
)

if (absl_FOUND)
  target_link_libraries(denox_onnx_proto
    PUBLIC
      absl::log          # brings log_internal_check_op, message, etc.
      absl::check        # ABSL_CHECK helpers
      absl::strings      # widely used by protobuf
      absl::cord         # protobuf uses Cord in newer versions
  )
else()
  find_library(ABSL_LOG        absl_log)
  find_library(ABSL_CHECK      absl_check)
  find_library(ABSL_STRINGS    absl_strings)
  find_library(ABSL_CORD       absl_cord)
  find_library(ABSL_RAW_LOG    absl_raw_logging_internal)
  find_library(ABSL_CHECK_OP   absl_log_internal_check_op)

  target_link_libraries(denox_onnx_proto
    PUBLIC
      ${ABSL_LOG} ${ABSL_CHECK} ${ABSL_STRINGS} ${ABSL_CORD}
      ${ABSL_RAW_LOG} ${ABSL_CHECK_OP}
  )
endif()

set_target_properties(denox_onnx_proto PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(denox_onnx_proto PUBLIC cxx_std_17)

add_library(denox::onnx_proto ALIAS denox_onnx_proto)
