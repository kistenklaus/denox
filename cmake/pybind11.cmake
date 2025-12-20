include_guard(GLOBAL)
include(${PROJECT_SOURCE_DIR}/cmake/colorful.cmake)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

find_package(pybind11 CONFIG)

if (pybind11_FOUND)
  log_success("✅ pybind11 available")
else()
  log_error("❌ pybind11 not found")
endif()

