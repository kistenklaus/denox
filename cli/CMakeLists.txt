

add_executable(denox EXCLUDE_FROM_ALL
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/cli/alloc/monotone_alloc.cpp

  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/cli/io/Pipe.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/cli/png/PngInputStream.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/cli/png/PngOutputStream.cpp

  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/cli/parser/lex/tokens/command.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/cli/parser/lex/tokens/literal.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/cli/parser/lex/tokens/option.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/cli/parser/lex/lex.cpp

  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/cli/parser/errors.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/cli/parser/parse.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/cli/parser/parse_artefact.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/cli/parser/parse_commands.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/cli/parser/parse_options.cpp

  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/cli/compile.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/cli/populate.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/cli/infer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/cli/bench.cpp

  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/cli/main.cpp
)


target_compile_features(denox PUBLIC cxx_std_20)
set_target_properties(denox PROPERTIES CXX_STANDARD 20)
set_target_properties(denox PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(denox PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(denox PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(denox 
  PUBLIC
    denox::onnx
    denox::dnx
    denox::db
    denox::png
    denox_compiler
    denox_common
    denox_spirv
    denox_database
    denox_runtime
)

if (NOT DENOX_DISABLE_WARNINGS)
  denox_enable_strict_warnings(denox)
endif()

if (DENOX_SAN) 
  target_compile_options(denox PRIVATE -fsanitize=undefined -fsanitize=address)
  target_link_options(denox PRIVATE -fsanitize=undefined -fsanitize=address)
endif()
