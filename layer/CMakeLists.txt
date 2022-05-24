cmake_minimum_required(VERSION 3.19)

target_include_directories(cppbp INTERFACE include)

target_sources(cppbp
        INTERFACE fully_connected.cpp
        INTERFACE input.cpp
        INTERFACE relu.cpp
        INTERFACE sigmoid.cpp
        INTERFACE softmax.cpp)


