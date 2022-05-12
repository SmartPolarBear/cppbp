cmake_minimum_required(VERSION 3.22)
project(cppbp)

set(CMAKE_CXX_STANDARD 23)

add_library(cppbp STATIC main.cpp)

add_subdirectory(libs)

target_link_libraries(cppbp
        INTERFACE Eigen3::Eigen)


add_subdirectory(base)
add_subdirectory(dataloader)
add_subdirectory(layer)
add_subdirectory(model)
add_subdirectory(utils)
add_subdirectory(tests)
add_subdirectory(optimizer)

add_subdirectory(examples)
