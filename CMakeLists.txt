cmake_minimum_required(VERSION 3.22)
project(cppbp)

set(CMAKE_CXX_STANDARD 20)

add_library(cppbp STATIC)

add_subdirectory(libs)

add_subdirectory(dataloader)
add_subdirectory(layer)
add_subdirectory(model)
add_subdirectory(utils)
add_subdirectory(tests)

add_subdirectory(examples)
