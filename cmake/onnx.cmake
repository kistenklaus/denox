include_guard(GLOBAL)
include(${PROJECT_SOURCE_DIR}/cmake/protobuf.cmake)

set(DENOX_ONNX_GENERATED_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)

add_library(denox_onnx)

protobuf_generate(
    TARGET denox_onnx
    PROTOS
        ${PROJECT_SOURCE_DIR}/third_party/onnx/onnx.proto
    IMPORT_DIRS
        ${PROJECT_SOURCE_DIR}/third_party/onnx
    PROTOC_OUT_DIR
        ${DENOX_ONNX_GENERATED_DIR}
)

target_include_directories(denox_onnx
    PUBLIC
        ${DENOX_ONNX_GENERATED_DIR}
)
