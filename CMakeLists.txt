cmake_minimum_required(VERSION 3.16)
project(MyTorchApp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use Release mode by default for better performance
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Torch REQUIRED)

add_executable(MyTorchApp main.cpp)
target_link_libraries(MyTorchApp "${TORCH_LIBRARIES}")

# Required to avoid missing symbols due to torch header-only parts
target_compile_definitions(MyTorchApp PRIVATE TORCH_API_INCLUDE_EXTENSION_H)
