include_guard(GLOBAL)
include(${PROJECT_SOURCE_DIR}/cmake/protobuf.cmake)

set(ONNX_PROTO ${PROJECT_SOURCE_DIR}/third_party/onnx/onnx.proto)
set(DENOX_ONNX_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(ONNX_PB_CC ${DENOX_ONNX_GENERATED_DIR}/onnx.pb.cc)
set(ONNX_PB_H  ${DENOX_ONNX_GENERATED_DIR}/onnx.pb.h)

file(MAKE_DIRECTORY ${DENOX_ONNX_GENERATED_DIR})

add_custom_command(
    OUTPUT
        ${ONNX_PB_CC}
        ${ONNX_PB_H}

    COMMAND ${PROTOC_EXEC}
        --cpp_out=${DENOX_ONNX_GENERATED_DIR}
        --proto_path=${PROJECT_SOURCE_DIR}/third_party/onnx
        ${ONNX_PROTO}

    DEPENDS
        ${ONNX_PROTO}
        protobuf::protoc

    COMMENT "Generating ONNX protobuf sources"
    VERBATIM
)

add_custom_target(denox_onnx_generate
    DEPENDS
        ${ONNX_PB_CC}
        ${ONNX_PB_H}
)

add_library(denox_onnx ${ONNX_PB_CC})
set_target_properties(denox_onnx PROPERTIES UNITY_BUILD OFF)

add_dependencies(denox_onnx denox_onnx_generate)

target_link_libraries(denox_onnx PUBLIC protobuf::libprotobuf)
target_include_directories(denox_onnx SYSTEM PUBLIC ${DENOX_ONNX_GENERATED_DIR})
