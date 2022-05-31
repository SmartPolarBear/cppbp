cmake_minimum_required(VERSION 3.19)

add_executable(mnist_bp main.cpp)

target_link_libraries(mnist_bp PRIVATE cppbp)
