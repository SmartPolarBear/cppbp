cmake_minimum_required(VERSION 3.19)

add_executable(iris_bp main.cpp)

target_link_libraries(iris_bp
        PRIVATE cppbp)
