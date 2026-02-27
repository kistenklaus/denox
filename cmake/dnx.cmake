include_guard(GLOBAL)

include(${PROJECT_SOURCE_DIR}/cmake/flatbuffers.cmake)

set(DNX_FBS ${PROJECT_SOURCE_DIR}/dnx.fbs)
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(GENERATED_HEADER ${GENERATED_DIR}/dnx.h)

file(MAKE_DIRECTORY ${GENERATED_DIR})

add_custom_command(
    OUTPUT ${GENERATED_HEADER}
    COMMAND ${FLATC_COMMAND}
            --cpp
            --no-emit-min-max-enum-values
            --filename-suffix ""
            -o ${GENERATED_DIR}
            ${DNX_FBS}
        DEPENDS ${DNX_FBS}
    COMMENT "Generating FlatBuffers schema"
    VERBATIM
)

add_custom_target(denox_dnx_generate
    DEPENDS ${GENERATED_HEADER}
)

add_library(denox_dnx INTERFACE)

add_dependencies(denox_dnx denox_dnx_generate)

target_include_directories(denox_dnx SYSTEM INTERFACE ${GENERATED_DIR})
target_link_libraries(denox_dnx INTERFACE flatbuffers)


